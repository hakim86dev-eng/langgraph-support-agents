"""Graph state. Plain ``TypedDict``s keep checkpoints JSON/msgpack-friendly and portable."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# typing_extensions.TypedDict is required by Pydantic on Python < 3.12.
from typing_extensions import TypedDict

from support_agents.tools.registry import WorkerName

RunStatus = Literal["approved", "needs_human_review"]


# ----------------------------------------------------------------- structured LLM outputs
class PlanStep(BaseModel):
    """One unit of work for a single worker."""

    worker: WorkerName = Field(description="Which worker handles this step.")
    instruction: str = Field(description="Self-contained instruction for the worker.")


class Plan(BaseModel):
    """The planner's decomposition of the customer request."""

    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""


class Review(BaseModel):
    """The reviewer's verdict on a draft reply."""

    approved: bool
    score: int = Field(ge=1, le=5, description="Overall quality from 1 (bad) to 5 (great).")
    feedback: str = Field(default="", description="Concrete changes needed if not approved.")


# ----------------------------------------------------------------- state records
class StepDict(TypedDict):
    id: int
    worker: WorkerName
    instruction: str


class ToolCallRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    output: str


class FindingDict(TypedDict):
    step_id: int
    worker: str
    instruction: str
    summary: str
    tool_calls: list[ToolCallRecord]


class ReviewDict(TypedDict):
    approved: bool
    score: int
    feedback: str


class TurnDict(TypedDict):
    question: str
    answer: str


class TraceDict(TypedDict):
    node: str
    duration_ms: float
    detail: str


def reset_or_extend(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Reducer: ``None`` clears the channel (new run), a list is appended."""
    if right is None:
        return []
    return [*(left or []), *right]


class AgentState(TypedDict, total=False):
    question: str
    customer_id: str | None
    plan: list[StepDict]
    findings: Annotated[list[FindingDict], reset_or_extend]
    draft: str
    review: ReviewDict | None
    revision: int
    final_answer: str | None
    status: RunStatus | None
    conversation: Annotated[list[TurnDict], operator.add]
    trace: Annotated[list[TraceDict], reset_or_extend]


class WorkerInput(TypedDict):
    step: StepDict
    question: str
    customer_id: str | None


def new_run_input(question: str, customer_id: str | None = None) -> dict[str, Any]:
    """Input for a new run on a (possibly existing) thread: clears per-run channels while the
    checkpointed ``conversation`` history is preserved."""
    return {
        "question": question,
        "customer_id": customer_id,
        "plan": [],
        "findings": None,
        "draft": "",
        "review": None,
        "revision": 0,
        "final_answer": None,
        "status": None,
        "trace": None,
    }
