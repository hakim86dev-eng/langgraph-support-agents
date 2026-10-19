"""Prompts for each agent. Every system prompt starts with a ``ROLE:`` tag (used for tracing
and by the offline demo/fake models to route deterministically)."""

from __future__ import annotations

from collections.abc import Sequence

from support_agents.graph.state import FindingDict, TurnDict
from support_agents.tools.registry import WORKER_DESCRIPTIONS, WorkerName

COMPANY = "Orbit Electronics"


def planner_system(max_steps: int) -> str:
    workers = "\n".join(f"- {name}: {desc}" for name, desc in WORKER_DESCRIPTIONS.items())
    return (
        "ROLE: planner\n"
        f"You plan work for the {COMPANY} customer-support team. Break the customer's request "
        f"into at most {max_steps} independent steps; each step is handled by one worker:\n"
        f"{workers}\n"
        "Steps run in parallel, so make each instruction self-contained and include any order "
        "or customer ids. Return an empty plan only for pure small talk. Call the Plan tool."
    )


def worker_system(worker: WorkerName) -> str:
    return (
        f"ROLE: worker:{worker}\n"
        f"You are the {worker} specialist at {COMPANY}: {WORKER_DESCRIPTIONS.get(worker, '')}. "
        "Use your tools to complete the instruction, then reply with a short factual summary of "
        "what you found. Never invent data that the tools did not return."
    )


WRITER_SYSTEM = (
    "ROLE: writer\n"
    f"You write replies to {COMPANY} customers. Use ONLY the numbered findings. Cite each "
    "finding you use as [n]. Be warm, concise and specific. If the findings are insufficient, "
    "say what is missing and offer to escalate. Never promise anything beyond policy."
)

REVIEWER_SYSTEM = (
    "ROLE: reviewer\n"
    "You are a strict QA reviewer for support replies. Approve only if the draft (1) answers "
    "every part of the request, (2) is fully grounded in the findings with [n] citations, and "
    "(3) is polite and makes no promises beyond policy. Call the Review tool."
)


def format_conversation(turns: Sequence[TurnDict], limit: int = 3) -> str:
    recent = list(turns)[-limit:]
    if not recent:
        return ""
    lines = [f"Customer: {t['question']}\nAgent: {t['answer']}" for t in recent]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def format_findings(findings: Sequence[FindingDict]) -> str:
    if not findings:
        return "FINDINGS:\n(none)"
    lines = [f"[{i}] ({f['worker']}) {f['summary']}" for i, f in enumerate(findings, start=1)]
    return "FINDINGS:\n" + "\n".join(lines)


def planner_user(question: str, customer_id: str | None, turns: Sequence[TurnDict]) -> str:
    return (
        f"{format_conversation(turns)}Customer id: {customer_id or 'unknown'}\n"
        f"Customer request: {question}"
    )


def worker_user(question: str, customer_id: str | None, instruction: str) -> str:
    return (
        f"Customer request: {question}\nCustomer id: {customer_id or 'unknown'}\n"
        f"Your instruction: {instruction}"
    )


def writer_user(
    question: str,
    findings: Sequence[FindingDict],
    previous_draft: str = "",
    feedback: str = "",
) -> str:
    text = f"Customer request: {question}\n{format_findings(findings)}"
    if previous_draft:
        text += f"\nPREVIOUS DRAFT:\n{previous_draft}\nREVIEWER FEEDBACK:\n{feedback}"
    return text


def reviewer_user(question: str, findings: Sequence[FindingDict], draft: str) -> str:
    return f"Customer request: {question}\n{format_findings(findings)}\nDRAFT REPLY:\n{draft}"
