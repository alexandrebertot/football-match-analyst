import copy
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletion

from football_agent.llm import MODEL


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


def text_reply(content: str | None, finish_reason: str = "stop") -> dict[str, Any]:
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
