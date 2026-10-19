from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_agents.graph import build_graph, new_run_input
from support_agents.llm import FakeToolCallingChatModel, ScriptedResponder
from support_agents.tools import Toolkit
from tests.conftest import plan, review

CFG: Any = {"configurable": {"thread_id": "persisted"}}


def _llm(answer: str) -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(
        responder=ScriptedResponder(
            {
                "planner": [plan()],
                "writer": [AIMessage(content=answer)],
                "reviewer": [review(True)],
            }
        )
    )


async def test_threads_survive_process_restart(tmp_path: Path, toolkit: Toolkit) -> None:
    db = str(tmp_path / "checkpoints.sqlite")

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(_llm("first answer"), toolkit, checkpointer=saver)
        await graph.ainvoke(new_run_input("first"), CFG)

    # A brand-new saver + graph (as after a restart) sees the same thread.
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(_llm("second answer"), toolkit, checkpointer=saver)
        snapshot = await graph.aget_state(CFG)
        assert snapshot.values["final_answer"] == "first answer"

        out = await graph.ainvoke(new_run_input("second"), CFG)
        assert [t["answer"] for t in out["conversation"]] == ["first answer", "second answer"]

        history = [s async for s in graph.aget_state_history(CFG)]
        assert len(history) > 5
        # Time travel: an earlier checkpoint still holds the first run's answer.
        assert any(s.values.get("final_answer") == "first answer" for s in history)
