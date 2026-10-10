import json
import sqlite3
from datetime import date

import httpx
import mlflow
import pytest
from mlflow.entities import SpanType, TraceState
from openai.types.chat import ChatCompletionMessageToolCall

from fakes import FakeLLM, text_reply, tool_reply
from football_agent.agent import MAX_TOOL_ROUNDS, answer, build_system_prompt, run_tool_call
from football_agent.data_api import BASE_URL, FootballDataClient
from football_agent.llm import MODEL
from football_agent.tools import TOOL_SCHEMAS, ToolContext

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


def tool_context(cache: sqlite3.Connection, status_code: int = 200) -> ToolContext:
    """Return a context whose fake football API always answers with the Ligue 1 standings."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=STANDINGS)

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handle))
    return ToolContext(football=FootballDataClient(http, cache), predictor=None, today=TODAY)


def make_tool_call(name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall.model_validate(
        tool_reply(name, arguments)["message"]["tool_calls"][0]
    )


def test_build_system_prompt_gives_today_weekday_and_date() -> None:
    assert "Today is Saturday 2026-10-03." in build_system_prompt(TODAY)


def test_run_tool_call_returns_tool_result_as_json(cache: sqlite3.Connection) -> None:
    tool_call = make_tool_call("get_standings", '{"competition": "FL1"}')

    result = run_tool_call(tool_context(cache), tool_call)

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

    result = run_tool_call(tool_context(cache, status_code), tool_call)

    assert result.startswith("Error: ")


def test_answer_returns_direct_reply_when_no_tool_is_needed(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([text_reply("Bonjour !")])

    reply = answer("Salut", llm, tool_context(cache))

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

    reply = answer("Qui est premier en Ligue 1 ?", llm, tool_context(cache))

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

    answer("Classement de la Ligue 1 ?", llm, tool_context(cache))

    assert llm.requests[1]["messages"][3]["content"].startswith("Error: ")


def test_answer_gives_up_after_max_tool_rounds(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([tool_reply("get_standings", '{"competition": "FL1"}')] * MAX_TOOL_ROUNDS)

    with pytest.raises(RuntimeError, match=f"No answer after {MAX_TOOL_ROUNDS} rounds"):
        answer("Classement ?", llm, tool_context(cache))

    assert len(llm.requests) == MAX_TOOL_ROUNDS


def test_answer_refuses_a_reply_cut_off_by_the_context_window(cache: sqlite3.Connection) -> None:
    llm = FakeLLM([text_reply("| 1 | PSG | 3 |\n| 2 | Bay", finish_reason="length")])

    with pytest.raises(RuntimeError, match="cut off"):
        answer("Classement complet de la C1 ?", llm, tool_context(cache))


@pytest.mark.parametrize("content", [None, "", " \n"], ids=["none", "empty", "blank"])
def test_answer_refuses_an_empty_reply(cache: sqlite3.Connection, content: str | None) -> None:
    llm = FakeLLM([text_reply(content)])

    with pytest.raises(RuntimeError, match="neither text nor a tool call"):
        answer("Classement ?", llm, tool_context(cache))


def last_trace() -> mlflow.entities.Trace:
    # Traces are written in the background: wait for the write before reading it back.
    mlflow.flush_trace_async_logging()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def test_answer_traces_the_question_each_tool_call_and_the_answer(
    tracing: None, cache: sqlite3.Connection
) -> None:
    llm = FakeLLM(
        [
            tool_reply("get_standings", '{"competition": "FL1"}'),
            text_reply("Monaco est premier."),
        ]
    )

    answer("Qui est premier en Ligue 1 ?", llm, tool_context(cache))

    trace = last_trace()
    spans = {span.name: span for span in trace.data.spans}
    assert spans["answer"].span_type == SpanType.AGENT
    assert spans["answer"].inputs == {"question": "Qui est premier en Ligue 1 ?"}
    assert spans["answer"].outputs == {"answer": "Monaco est premier."}
    tool_span = spans["get_standings"]
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.parent_id == spans["answer"].span_id
    assert tool_span.inputs == {"arguments": '{"competition": "FL1"}'}
    assert "Monaco" in tool_span.outputs["result"]


def test_answer_trace_is_marked_as_failed_when_the_agent_gives_up(
    tracing: None, cache: sqlite3.Connection
) -> None:
    llm = FakeLLM([text_reply("")])

    with pytest.raises(RuntimeError):
        answer("Classement ?", llm, tool_context(cache))

    trace = last_trace()
    assert trace.info.state == TraceState.ERROR
