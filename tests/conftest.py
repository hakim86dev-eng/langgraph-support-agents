from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from support_agents.llm.fake import ToolSpec, tool_call_message
from support_agents.tools import Toolkit, build_toolkit

TODAY = date(2026, 10, 19)


@pytest.fixture
def toolkit() -> Toolkit:
    return build_toolkit(today=lambda: TODAY)


def plan(*steps: tuple[str, str]) -> AIMessage:
    return tool_call_message(
        "Plan", {"steps": [{"worker": w, "instruction": i} for w, i in steps], "rationale": "t"}
    )


def review(approved: bool, feedback: str = "", score: int | None = None) -> AIMessage:
    return tool_call_message(
        "Review",
        {"approved": approved, "score": score or (5 if approved else 2), "feedback": feedback},
    )


def tool_then_summarise(name: str, args: dict[str, Any]):  # type: ignore[no-untyped-def]
    """Worker script: call ``name`` once, then summarise with the raw tool output."""

    def respond(messages: list[BaseMessage], tools: list[ToolSpec]) -> AIMessage:
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"summary: {messages[-1].content}")
        return tool_call_message(name, args)

    return respond
