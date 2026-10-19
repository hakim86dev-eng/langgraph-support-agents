from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agents.graph import build_graph, new_run_input
from support_agents.llm import FakeToolCallingChatModel, ScriptedResponder, tool_call_message
from support_agents.llm.demo import demo_responder
from support_agents.tools import Toolkit
from tests.conftest import plan, review, tool_then_summarise

CFG: Any = {"configurable": {"thread_id": "t"}}


def make(script: dict[str, list[Any]], toolkit: Toolkit, **kwargs: Any):  # type: ignore[no-untyped-def]
    llm = FakeToolCallingChatModel(responder=ScriptedResponder(script))
    graph = build_graph(llm, toolkit, checkpointer=InMemorySaver(), **kwargs)
    return graph, llm


async def test_happy_path_runs_workers_in_parallel_and_approves(toolkit: Toolkit) -> None:
    graph, llm = make(
        {
            "planner": [plan(("account", "Look up order ORD-1001"), ("knowledge", "returns"))],
            "worker:account": [tool_then_summarise("get_order", {"order_id": "ORD-1001"})],
            "worker:knowledge": [tool_then_summarise("search_knowledge_base", {"query": "return"})],
            "writer": [AIMessage(content="You can return it [1][2].")],
            "reviewer": [review(True)],
        },
        toolkit,
    )
    out = await graph.ainvoke(new_run_input("Can I return ORD-1001?", "CUST-1"), CFG)

    assert out["status"] == "approved"
    assert out["final_answer"] == "You can return it [1][2]."
    assert [f["worker"] for f in sorted(out["findings"], key=lambda f: f["step_id"])] == [
        "account",
        "knowledge",
    ]
    account = next(f for f in out["findings"] if f["worker"] == "account")
    assert account["tool_calls"][0]["tool"] == "get_order"
    assert '"days_since_delivery": 18' in account["tool_calls"][0]["output"]
    # The writer saw both findings, numbered.
    writer_prompt = llm.calls_for("writer")[0][-1].content
    assert "[1] (account) summary:" in writer_prompt
    assert "[2] (knowledge) summary:" in writer_prompt
    assert out["trace"][0]["node"] == "planner"
    assert out["conversation"] == [
        {"question": "Can I return ORD-1001?", "answer": "You can return it [1][2]."}
    ]


async def test_reviewer_feedback_triggers_revision(toolkit: Toolkit) -> None:
    graph, llm = make(
        {
            "planner": [plan(("math", "Compute 2*3"))],
            "worker:math": [tool_then_summarise("calculate", {"expression": "2*3"})],
            "writer": [AIMessage(content="draft one"), AIMessage(content="draft two [1]")],
            "reviewer": [review(False, "Add a citation."), review(True)],
        },
        toolkit,
    )
    out = await graph.ainvoke(new_run_input("2*3?"), CFG)
    assert out["status"] == "approved"
    assert out["final_answer"] == "draft two [1]"
    assert out["revision"] == 2
    second_prompt = llm.calls_for("writer")[1][-1].content
    assert "PREVIOUS DRAFT:\ndraft one" in second_prompt
    assert "Add a citation." in second_prompt


async def test_revision_budget_escalates_to_human(toolkit: Toolkit) -> None:
    graph, llm = make(
        {
            "planner": [plan()],
            "writer": [AIMessage(content="meh")],
            "reviewer": [review(False, "Still wrong.")],
        },
        toolkit,
        max_revisions=1,
    )
    out = await graph.ainvoke(new_run_input("help"), CFG)
    assert out["status"] == "needs_human_review"
    assert out["final_answer"] == "meh"
    assert len(llm.calls_for("writer")) == 2  # first draft + one revision
    assert out["plan"] == []  # empty plan skips workers
    assert out["findings"] == []


async def test_planner_failure_falls_back_to_knowledge_search(toolkit: Toolkit) -> None:
    graph, _ = make(
        {
            "planner": [tool_call_message("Plan", {"steps": [{"worker": "hacker"}]})],
            "worker:knowledge": [tool_then_summarise("search_knowledge_base", {"query": "ship"})],
            "writer": [AIMessage(content="ok [1]")],
            "reviewer": [review(True)],
        },
        toolkit,
    )
    out = await graph.ainvoke(new_run_input("shipping?"), CFG)
    assert out["plan"] == [{"id": 1, "worker": "knowledge", "instruction": "shipping?"}]
    assert out["trace"][0]["detail"].startswith("fallback plan")
    assert out["status"] == "approved"


async def test_reviewer_failure_does_not_loop(toolkit: Toolkit) -> None:
    graph, llm = make(
        {
            "planner": [plan()],
            "writer": [AIMessage(content="draft")],
            "reviewer": [AIMessage(content="not a tool call")],
        },
        toolkit,
    )
    out = await graph.ainvoke(new_run_input("hi"), CFG)
    assert out["status"] == "needs_human_review"
    assert out["review"]["feedback"].startswith("review failed")
    assert len(llm.calls_for("writer")) == 1


async def test_worker_handles_unknown_tool_and_tool_budget(toolkit: Toolkit) -> None:
    graph, _ = make(
        {
            "planner": [plan(("math", "loop forever"))],
            "worker:math": [tool_call_message("rm_rf", {"path": "/"})],
            "writer": [AIMessage(content="sorry")],
            "reviewer": [review(True)],
        },
        toolkit,
        worker_max_iterations=2,
    )
    out = await graph.ainvoke(new_run_input("x"), CFG)
    (finding,) = out["findings"]
    assert finding["summary"].startswith("Step incomplete: tool-call budget exhausted")
    assert [c["output"] for c in finding["tool_calls"]] == ["error: unknown tool 'rm_rf'"] * 2


async def test_plan_is_capped_at_max_steps(toolkit: Toolkit) -> None:
    steps = [("math", f"Compute {i}+1") for i in range(6)]
    graph, llm = make(
        {
            "planner": [plan(*steps)],
            "worker:math": [AIMessage(content="done")],
            "writer": [AIMessage(content="ok")],
            "reviewer": [review(True)],
        },
        toolkit,
        max_plan_steps=3,
    )
    out = await graph.ainvoke(new_run_input("many"), CFG)
    assert len(out["plan"]) == 3
    assert len(out["findings"]) == 3
    assert len(llm.calls_for("worker:math")) == 3


async def test_new_run_resets_per_run_state_but_keeps_conversation(toolkit: Toolkit) -> None:
    graph, llm = make(
        {
            "planner": [plan(("math", "1+1")), plan()],
            "worker:math": [AIMessage(content="two")],
            "writer": [AIMessage(content="first [1]"), AIMessage(content="second")],
            "reviewer": [review(True)],
        },
        toolkit,
    )
    await graph.ainvoke(new_run_input("first question"), CFG)
    out = await graph.ainvoke(new_run_input("follow-up"), CFG)
    assert out["findings"] == []
    assert out["final_answer"] == "second"
    assert [t["question"] for t in out["conversation"]] == ["first question", "follow-up"]
    # The planner of the second run sees the earlier turn.
    assert "Customer: first question" in llm.calls_for("planner")[1][-1].content


async def test_demo_model_end_to_end(toolkit: Toolkit) -> None:
    llm = FakeToolCallingChatModel(responder=demo_responder)
    graph = build_graph(llm, toolkit, checkpointer=InMemorySaver())
    out = await graph.ainvoke(
        new_run_input("Can I still return my earbuds from order ORD-1001?", "CUST-1"), CFG
    )
    assert out["status"] == "approved"
    assert {f["worker"] for f in out["findings"]} == {"account", "knowledge"}
    assert "18 days ago" in out["final_answer"]
    assert "14 days" in out["final_answer"]

    small_talk = await graph.ainvoke(new_run_input("hello"), CFG)
    assert small_talk["plan"] == []
    assert "order number" in small_talk["final_answer"]

    maths = await graph.ainvoke(new_run_input("What is 129 * 0.9 after my discount?"), CFG)
    assert "116.1" in maths["final_answer"]
