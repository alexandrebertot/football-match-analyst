import copy
import json
import sqlite3
from datetime import date
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from football_agent.agent import MAX_TOOL_ROUNDS, answer, build_system_prompt, run_tool_call
from football_agent.data_api import BASE_URL, FootballDataClient
from football_agent.llm import MODEL
from football_agent.tools import TOOL_SCHEMAS

TODAY = date(2026, 10, 3)

STANDINGS = {
    "competition": {"code": "FL1", "name": "Ligue 1"},
    "season": {"startDate": "2026-08-22"},
    "standings": [
        {
            "table": [
                {
                    "position": 1,
                    "team": {"id": 548, "name": "AS Monaco FC", "shortName": "Monaco"},
                    "playedGames": 5,
                    "won": 4,
                    "draw": 1,
                    "lost": 0,
                    "points": 13,
                    "goalsFor": 8,
                    "goalsAgainst": 3,
                    "goalDifference": 5,
                }
            ]
        }
    ],
}


class FakeLLM:
    """Stand-in for the OpenAI client: returns scripted choices, records each request."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = iter(replies)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> ChatCompletion:
        # The agent keeps appending to the same messages list, so keep a snapshot of this call.
        self.requests.append(copy.deepcopy(request))
        return ChatCompletion.model_validate(
            {
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [{"index": 0, **next(self.replies)}],
            }
        )


def text_reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}


def tool_reply(name: str, arguments: str) -> dict[str, Any]:
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return {
        "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
    }


def football_api(cache: sqlite3.Connection, status_code: int = 200) -> FootballDataClient:
    """Return a football client whose fake API always answers with the Ligue 1 standings."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=STANDINGS)

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))
    return FootballDataClient(http, cache)


def make_tool_call(name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall.model_validate(
        tool_reply(name, arguments)["message"]["tool_calls"][0]
    )


def test_build_system_prompt_gives_today_weekday_and_date() -> None:
    assert "Today is Saturday 2026-10-03." in build_system_prompt(TODAY)


def test_run_tool_call_returns_tool_result_as_json(cache: sqlite3.Connection) -> None:
    tool_call = make_tool_call("get_standings", '{"competition": "FL1"}')

    result = run_tool_call(football_api(cache), tool_call)

    assert json.loads(result)["table"][0]["team"] == "Monaco"


@pytest.mark.parametrize(
    ("arguments", "status_code"),
    [
        ('{"competition": ', 200),
        ('{"league": "FL1"}', 200),
        ('{"competition": "FL1", "season": "2025-26"}', 400),
    ],
    ids=["invalid-json", "wrong-argument-name", "rejected-by-api"],
)
def test_run_tool_call_turns_fixable_errors_into_messages(
    cache: sqlite3.Connection, arguments: str, status_code: int
) -> None:
    tool_call = make_tool_call("get_standings", arguments)

    result = run_tool_call(football_api(cache, status_code), tool_call)

    assert result.startswith("Error: ")


def test_answer_returns_direct_reply_when_no_tool_is_needed(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([text_reply("Bonjour !")])

    reply = answer("Salut", llm, football_api(cache), TODAY)

    assert reply == "Bonjour !"
    request = llm.requests[0]
    assert request["model"] == MODEL
    assert request["tools"] == TOOL_SCHEMAS
    assert [message["role"] for message in request["messages"]] == ["system", "user"]


def test_answer_runs_requested_tool_and_sends_result_back(cache: sqlite3.Connection) -> None:
    llm = FakeLLM(
        [
            tool_reply("get_standings", '{"competition": "FL1"}'),
            text_reply("Monaco est premier."),
        ]
    )

    reply = answer("Qui est premier en Ligue 1 ?", llm, football_api(cache), TODAY)

    assert reply == "Monaco est premier."
    messages = llm.requests[1]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_standings"
    assert messages[3]["tool_call_id"] == "call_1"
    assert "Monaco" in messages[3]["content"]


def test_answer_sends_tool_errors_back_to_the_llm(cache: sqlite3.Connection) -> None:
    llm = FakeLLM(
        [
            tool_reply("get_standings", '{"league": "FL1"}'),
            text_reply("Je réessaie."),
        ]
    )

    answer("Classement de la Ligue 1 ?", llm, football_api(cache), TODAY)

    assert llm.requests[1]["messages"][3]["content"].startswith("Error: ")


def test_answer_gives_up_after_max_tool_rounds(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([tool_reply("get_standings", '{"competition": "FL1"}')] * MAX_TOOL_ROUNDS)

    with pytest.raises(RuntimeError, match=f"No answer after {MAX_TOOL_ROUNDS} rounds"):
        answer("Classement ?", llm, football_api(cache), TODAY)

    assert len(llm.requests) == MAX_TOOL_ROUNDS


def test_answer_refuses_a_reply_cut_off_by_the_context_window(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([text_reply("| 1 | PSG | 3 |\n| 2 | Bay", finish_reason="length")])

    with pytest.raises(RuntimeError, match="cut off"):
        answer("Classement complet de la C1 ?", llm, football_api(cache), TODAY)
