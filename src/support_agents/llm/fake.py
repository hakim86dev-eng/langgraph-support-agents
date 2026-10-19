"""A fake LangChain chat model that supports tool calling and structured output.

``BaseChatModel.with_structured_output`` is implemented on top of ``bind_tools`` in
langchain-core, so supporting ``bind_tools`` gives us both: tests can script plans, tool calls
and reviews exactly as a real provider would return them, with zero network access.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

ToolSpec = dict[str, Any]
Responder = Callable[[Sequence[BaseMessage], Sequence[ToolSpec]], AIMessage]

_call_ids = itertools.count(1)


def tool_call_message(name: str, args: Mapping[str, Any], content: str = "") -> AIMessage:
    """Build an ``AIMessage`` requesting a single tool call."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": dict(args), "id": f"call_{next(_call_ids)}"}],
    )


def role_of(messages: Sequence[BaseMessage]) -> str:
    """Return the ``ROLE:`` tag of the first system message (e.g. ``planner``)."""
    for message in messages:
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            first = message.content.splitlines()[0] if message.content else ""
            if first.startswith("ROLE:"):
                return first.removeprefix("ROLE:").strip()
    return ""


class FakeToolCallingChatModel(BaseChatModel):
    """Delegates every call to ``responder(messages, bound_tools)`` and records the call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder
    calls: list[tuple[str, list[BaseMessage], list[ToolSpec]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[ToolSpec] = list(kwargs.get("tools") or [])
        self.calls.append((role_of(messages), list(messages), tools))
        message = self.responder(messages, tools)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def calls_for(self, role: str) -> list[list[BaseMessage]]:
        return [messages for r, messages, _ in self.calls if r == role]


class ScriptedResponder:
    """Returns queued responses per role; the last response for a role repeats.

    Routing by role (not call order) keeps tests deterministic even though workers run in
    parallel. Values may be messages or callables computing a message from the inputs.
    """

    def __init__(
        self,
        script: Mapping[str, Sequence[AIMessage | Responder]],
        default: AIMessage | None = None,
    ) -> None:
        self._script = {role: list(items) for role, items in script.items()}
        self._cursor: dict[str, int] = {}
        self._default = default
        self._lock = threading.Lock()

    def __call__(self, messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage:
        role = role_of(messages)
        with self._lock:
            items = self._script.get(role)
            if not items:
                if self._default is None:
                    raise KeyError(f"no scripted response for role {role!r}")
                return self._default
            index = self._cursor.get(role, 0)
            self._cursor[role] = index + 1
            item = items[min(index, len(items) - 1)]
        return item(messages, tools) if callable(item) else item
