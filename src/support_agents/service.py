"""Application service around the compiled graph: runs, streaming, state and history."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from support_agents.graph.state import (
    FindingDict,
    ReviewDict,
    StepDict,
    TraceDict,
    TurnDict,
    new_run_input,
)


class ThreadBusyError(RuntimeError):
    """Raised when a run is already in progress on the thread."""


class RunResult(BaseModel):
    thread_id: str
    status: str
    answer: str
    plan: list[StepDict]
    findings: list[FindingDict]
    review: ReviewDict | None
    revisions: int
    trace: list[TraceDict]


class ThreadState(BaseModel):
    thread_id: str
    conversation: list[TurnDict]
    last_run: RunResult | None


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    created_at: str | None


def _to_result(thread_id: str, values: dict[str, Any]) -> RunResult:
    return RunResult(
        thread_id=thread_id,
        status=values.get("status") or "incomplete",
        answer=values.get("final_answer") or "",
        plan=values.get("plan", []),
        findings=sorted(values.get("findings", []), key=lambda f: f["step_id"]),
        review=values.get("review"),
        revisions=values.get("revision", 0),
        trace=values.get("trace", []),
    )


class AgentService:
    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any], recursion_limit: int = 50):
        self.graph = graph
        self.recursion_limit = recursion_limit
        self._locks: dict[str, asyncio.Lock] = {}

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise ThreadBusyError(thread_id)
        return lock

    async def run(self, thread_id: str, question: str, customer_id: str | None) -> RunResult:
        async with self._lock(thread_id):
            values = await self.graph.ainvoke(
                new_run_input(question, customer_id), self._config(thread_id)
            )
        return _to_result(thread_id, values)

    async def stream(
        self, thread_id: str, question: str, customer_id: str | None
    ) -> AsyncIterator[str]:
        """Yield Server-Sent Events: one per node update, then a final ``done`` event."""
        lock = self._lock(thread_id)
        async with lock:
            async for chunk in self.graph.astream(
                new_run_input(question, customer_id),
                self._config(thread_id),
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    yield f"event: {node}\ndata: {json.dumps(update, default=str)}\n\n"
        result = await self.get_last_run(thread_id)
        payload = result.model_dump_json() if result else "{}"
        yield f"event: done\ndata: {payload}\n\n"

    async def _values(self, thread_id: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return dict(snapshot.values or {})

    async def get_last_run(self, thread_id: str) -> RunResult | None:
        values = await self._values(thread_id)
        return _to_result(thread_id, values) if values else None

    async def get_thread(self, thread_id: str) -> ThreadState | None:
        values = await self._values(thread_id)
        if not values:
            return None
        return ThreadState(
            thread_id=thread_id,
            conversation=values.get("conversation", []),
            last_run=_to_result(thread_id, values),
        )

    async def history(self, thread_id: str, limit: int = 50) -> list[CheckpointInfo]:
        items: list[CheckpointInfo] = []
        async for snap in self.graph.aget_state_history(self._config(thread_id), limit=limit):
            meta = snap.metadata or {}
            items.append(
                CheckpointInfo(
                    checkpoint_id=str(snap.config["configurable"]["checkpoint_id"]),
                    step=int(meta.get("step", -1)),
                    source=str(meta.get("source", "")),
                    next=list(snap.next),
                    created_at=snap.created_at,
                )
            )
        return items
