from __future__ import annotations

import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from support_agents.config import Settings
from support_agents.graph.state import Plan
from support_agents.llm import (
    FakeToolCallingChatModel,
    ScriptedResponder,
    create_chat_model,
    role_of,
    tool_call_message,
)
from support_agents.tools import calculate

SYS = SystemMessage("ROLE: planner\nYou plan.")


def test_role_of() -> None:
    assert role_of([SYS, HumanMessage("hi")]) == "planner"
    assert role_of([HumanMessage("hi")]) == ""


def test_structured_output_via_tool_calls() -> None:
    msg = tool_call_message("Plan", {"steps": [{"worker": "math", "instruction": "1+1"}]})
    model = FakeToolCallingChatModel(responder=ScriptedResponder({"planner": [msg]}))
    result = model.with_structured_output(Plan).invoke([SYS, HumanMessage("q")])
    assert isinstance(result, Plan)
    assert result.steps[0].worker == "math"
    role, _, tools = model.calls[0]
    assert role == "planner"
    assert tools[0]["function"]["name"] == "Plan"


def test_bind_tools_passes_tool_schemas() -> None:
    model = FakeToolCallingChatModel(responder=lambda m, t: AIMessage(content=str(len(t))))
    out = model.bind_tools([calculate]).invoke([HumanMessage("x")])
    assert out.content == "1"
    assert model.calls[0][2][0]["function"]["name"] == "calculate"


def test_scripted_responder_sequence_default_and_errors() -> None:
    responder = ScriptedResponder(
        {"planner": [AIMessage(content="a"), lambda m, t: AIMessage(content="b")]},
        default=None,
    )
    contents = [responder([SYS], []).content for _ in range(3)]
    assert contents == ["a", "b", "b"]  # last item repeats
    with pytest.raises(KeyError):
        responder([SystemMessage("ROLE: writer")], [])
    with_default = ScriptedResponder({}, default=AIMessage(content="d"))
    assert with_default([SYS], []).content == "d"


def test_factory_demo_and_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(create_chat_model(Settings(_env_file=None)), FakeToolCallingChatModel)
    for provider, module in [
        ("openai", "langchain_openai"),
        ("anthropic", "langchain_anthropic"),
        ("ollama", "langchain_ollama"),
    ]:
        monkeypatch.setitem(sys.modules, module, None)  # force ImportError
        with pytest.raises(RuntimeError, match=rf"\[{provider}\]"):
            create_chat_model(Settings(llm_provider=provider, _env_file=None))
