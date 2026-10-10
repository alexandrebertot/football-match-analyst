import re
from dataclasses import dataclass
from typing import Any

from football_agent.tools import normalize_name

# LLMs often write typographic dashes and apostrophes where the tools return plain ones.
PLAIN_CHARACTERS = str.maketrans({"–": "-", "—": "-", "−": "-", "’": "'", "‘": "'"})


@dataclass(frozen=True)
class Fact:
    """Something the answer must state, with every spelling accepted for it."""

    description: str
    spellings: tuple[str, ...]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


def normalize_text(text: str) -> str:
    """Make spellings comparable: '57,9 %' and '57.9%', '2 – 1' and '2:1' and '2-1',
    'M’gladbach' and "M'gladbach", 'Étienne' and 'etienne'."""
    text = normalize_name(text).translate(PLAIN_CHARACTERS)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s*:\s*(\d)", r"\1-\2", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+%", "%", text)
    return re.sub(r"\s+", " ", text)


def fact_found(fact: Fact, answer: str) -> bool:
    """Whether the answer states the fact in one of its accepted spellings, as a whole word:
    '15' must not match inside '2015', '1.5' or '15.5'."""
    text = normalize_text(answer)
    # [^\W_] is a letter or a digit: unlike \w it excludes "_", so Markdown italics like
    # "_Lens_" still count as the whole word "lens".
    return any(
        re.search(rf"(?<![^\W_])(?<!\.){re.escape(normalize_text(spelling))}(?![^\W_]|\.\d)", text)
        for spelling in fact.spellings
    )


def called_expected_tool(calls: list[ToolCall], tools: tuple[str, ...], competition: str) -> bool:
    """Whether one of the tools that can answer the question was called on the right competition."""
    return any(
        call.name in tools and call.arguments.get("competition") == competition for call in calls
    )


def score_case(
    answer: str,
    calls: list[ToolCall],
    tools: tuple[str, ...],
    competition: str,
    facts: list[Fact],
) -> dict[str, Any]:
    """Score one answer: right tool, facts stated, and tool calls that failed along the way.

    A failed call that the agent then fixed does not make the answer fail.
    """
    found = sum(fact_found(fact, answer) for fact in facts)
    right_tool = called_expected_tool(calls, tools, competition)
    return {
        "right_tool": right_tool,
        "facts_found": found,
        "facts_expected": len(facts),
        "failed_calls": sum(call.result.startswith("Error:") for call in calls),
        "success": right_tool and found == len(facts),
    }
