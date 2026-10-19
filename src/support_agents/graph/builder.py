"""LangGraph wiring: planner -> parallel tool-using workers -> writer <-> reviewer -> finalize.

START -> planner --(Send per step)--> worker* --> writer --> reviewer --> finalize -> END
              \\__________(empty plan)_________/        ^        |
                                                       \\_revise_/
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from support_agents.graph import prompts
from support_agents.graph.state import (
    AgentState,
    FindingDict,
    Plan,
    PlanStep,
    Review,
    StepDict,
    ToolCallRecord,
    TraceDict,
    TurnDict,
    WorkerInput,
)
from support_agents.tools.registry import Toolkit

logger = logging.getLogger(__name__)


def _trace(node: str, started: float, detail: str = "") -> list[TraceDict]:
    return [
        {
            "node": node,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": detail,
        }
    ]


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
    return "".join(parts).strip()


def build_graph(
    llm: BaseChatModel,
    toolkit: Toolkit,
    *,
    max_plan_steps: int = 4,
    max_revisions: int = 2,
    worker_max_iterations: int = 4,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    planner_llm = llm.with_structured_output(Plan)
    reviewer_llm = llm.with_structured_output(Review)

    # ------------------------------------------------------------------ planner
    async def planner(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        try:
            plan = await planner_llm.ainvoke(
                [
                    SystemMessage(prompts.planner_system(max_plan_steps)),
                    HumanMessage(
                        prompts.planner_user(
                            question, state.get("customer_id"), state.get("conversation", [])
                        )
                    ),
                ]
            )
            if not isinstance(plan, Plan):
                raise TypeError(f"planner returned {type(plan).__name__}")
            detail = f"{len(plan.steps)} step(s)"
        except Exception as exc:  # malformed structured output, provider error, ...
            logger.warning("planner failed, falling back to a knowledge search: %s", exc)
            plan = Plan(steps=[PlanStep(worker="knowledge", instruction=question)])
            detail = f"fallback plan ({type(exc).__name__})"
        steps: list[StepDict] = [
            {"id": i, "worker": s.worker, "instruction": s.instruction}
            for i, s in enumerate(plan.steps[:max_plan_steps], start=1)
        ]
        return {"plan": steps, "trace": _trace("planner", started, detail)}

    def dispatch(state: AgentState) -> list[Send] | Literal["writer"]:
        if not state.get("plan"):
            return "writer"
        return [
            Send(
                "worker",
                WorkerInput(
                    step=step, question=state["question"], customer_id=state.get("customer_id")
                ),
            )
            for step in state["plan"]
        ]

    # ------------------------------------------------------------------ workers
    async def worker(payload: WorkerInput) -> dict[str, Any]:
        started = time.perf_counter()
        step = payload["step"]
        tools = toolkit.get(step["worker"], [])
        by_name = {t.name: t for t in tools}
        model = llm.bind_tools(tools) if tools else llm
        messages: list[BaseMessage] = [
            SystemMessage(prompts.worker_system(step["worker"])),
            HumanMessage(
                prompts.worker_user(
                    payload["question"], payload["customer_id"], step["instruction"]
                )
            ),
        ]
        records: list[ToolCallRecord] = []
        summary = ""
        for _ in range(worker_max_iterations):
            response = await model.ainvoke(messages)
            messages.append(response)
            calls = response.tool_calls if isinstance(response, AIMessage) else []
            if not calls:
                summary = _text(response)
                break
            for call in calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    output = f"error: unknown tool '{call['name']}'"
                else:
                    try:
                        output = str(await tool.ainvoke(call["args"]))
                    except Exception as exc:  # tool errors are reported back to the model
                        output = f"error: {exc}"
                records.append({"tool": call["name"], "args": call["args"], "output": output})
                messages.append(ToolMessage(content=output, tool_call_id=call["id"] or ""))
        else:
            summary = "Step incomplete: tool-call budget exhausted. " + (
                records[-1]["output"][:300] if records else ""
            )
        finding: FindingDict = {
            "step_id": step["id"],
            "worker": step["worker"],
            "instruction": step["instruction"],
            "summary": summary or "No result.",
            "tool_calls": records,
        }
        return {
            "findings": [finding],
            "trace": _trace(f"worker:{step['worker']}", started, f"{len(records)} tool call(s)"),
        }

    # ------------------------------------------------------------------ writer
    async def writer(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        findings = sorted(state.get("findings", []), key=lambda f: f["step_id"])
        review = state.get("review")
        user = prompts.writer_user(
            state["question"],
            findings,
            previous_draft=state.get("draft", "") if review else "",
            feedback=review["feedback"] if review else "",
        )
        response = await llm.ainvoke([SystemMessage(prompts.WRITER_SYSTEM), HumanMessage(user)])
        revision = state.get("revision", 0) + 1
        return {
            "draft": _text(response),
            "revision": revision,
            "trace": _trace("writer", started, f"draft {revision}"),
        }

    # ------------------------------------------------------------------ reviewer
    async def reviewer(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        findings = sorted(state.get("findings", []), key=lambda f: f["step_id"])
        try:
            verdict = await reviewer_llm.ainvoke(
                [
                    SystemMessage(prompts.REVIEWER_SYSTEM),
                    HumanMessage(
                        prompts.reviewer_user(state["question"], findings, state["draft"])
                    ),
                ]
            )
            if not isinstance(verdict, Review):
                raise TypeError(f"reviewer returned {type(verdict).__name__}")
        except Exception as exc:
            logger.warning("reviewer failed; escalating to a human: %s", exc)
            verdict = Review(approved=False, score=1, feedback=f"review failed: {exc}")
            # Do not loop on a broken reviewer: jump straight to finalize.
            return {
                "review": verdict.model_dump(),
                "revision": max_revisions + 1,
                "trace": _trace("reviewer", started, "review failed"),
            }
        detail = "approved" if verdict.approved else f"changes requested (score {verdict.score})"
        return {"review": verdict.model_dump(), "trace": _trace("reviewer", started, detail)}

    def after_review(state: AgentState) -> Literal["writer", "finalize"]:
        review = state.get("review")
        if (review and review["approved"]) or state.get("revision", 0) > max_revisions:
            return "finalize"
        return "writer"

    # ------------------------------------------------------------------ finalize
    def finalize(state: AgentState) -> dict[str, Any]:
        review = state.get("review")
        approved = bool(review and review["approved"])
        turn: TurnDict = {"question": state["question"], "answer": state["draft"]}
        return {
            "final_answer": state["draft"],
            "status": "approved" if approved else "needs_human_review",
            "conversation": [turn],
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    # Workers receive a WorkerInput payload via Send rather than the full AgentState.
    graph.add_node("worker", cast(Any, worker))
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch, ["worker", "writer"])
    graph.add_edge("worker", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", after_review, ["writer", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
