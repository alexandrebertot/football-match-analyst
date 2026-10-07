import json
from datetime import date
from typing import Any

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from football_agent.llm import MODEL
from football_agent.tools import TOOL_SCHEMAS, WEEKDAYS, ToolContext, call_tool

MAX_TOOL_ROUNDS = 5


def build_system_prompt(today: date) -> str:
    return (
        "You are a football assistant for the top 5 European leagues and the Champions League. "
        f"Today is {WEEKDAYS[today.weekday()]} {today.isoformat()}. "
        "Always use the tools to get scores, fixtures, standings, team form and predictions; "
        "never answer them from memory. "
        "If a tool returns an error, fix the arguments and try again. "
        "Answer in the language of the question."
    )


def run_tool_call(context: ToolContext, tool_call: ChatCompletionMessageToolCall) -> str:
    """Run one tool call and return its result as JSON, or an error message the LLM can act on."""
    try:
        arguments = json.loads(tool_call.function.arguments)
        result = call_tool(context, tool_call.function.name, arguments)
    except (json.JSONDecodeError, ValueError, TypeError, httpx.HTTPError) as error:
        return f"Error: {error}"
    return json.dumps(result, ensure_ascii=False)


def answer(question: str, llm: OpenAI, context: ToolContext) -> str:
    """Answer `question`, letting the LLM call the football tools until it replies with text."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(context.today)},
        {"role": "user", "content": question},
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("The LLM reply was cut off because its context window is full.")
        message = choice.message
        if not message.tool_calls:
            if not message.content or not message.content.strip():
                raise RuntimeError("The LLM replied with neither text nor a tool call.")
            return message.content
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": run_tool_call(context, tool_call),
                }
            )
    raise RuntimeError(f"No answer after {MAX_TOOL_ROUNDS} rounds of tool calls.")
