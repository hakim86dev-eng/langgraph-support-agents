"""Which tools each worker agent may use."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Literal, get_args

from langchain_core.tools import BaseTool

from support_agents.tools.calculator import calculate
from support_agents.tools.crm import CRM, make_crm_tools
from support_agents.tools.knowledge_base import KnowledgeBase, make_kb_tools

WorkerName = Literal["knowledge", "account", "math"]
WORKERS: tuple[WorkerName, ...] = get_args(WorkerName)
Toolkit = dict[WorkerName, list[BaseTool]]

WORKER_DESCRIPTIONS: dict[WorkerName, str] = {
    "knowledge": "searches help-centre policies (returns, refunds, shipping, warranty, payments)",
    "account": "looks up customer profiles and orders in the CRM",
    "math": "performs arithmetic such as refund amounts, discounts and totals",
}


def build_toolkit(
    kb: KnowledgeBase | None = None,
    crm: CRM | None = None,
    today: Callable[[], date] = date.today,
) -> Toolkit:
    return {
        "knowledge": make_kb_tools(kb or KnowledgeBase.bundled()),
        "account": make_crm_tools(crm or CRM.bundled(), today),
        "math": [calculate],
    }
