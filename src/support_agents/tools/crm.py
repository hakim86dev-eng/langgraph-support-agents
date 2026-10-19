"""A read-only fake CRM (customers and orders), exposed as LangChain tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from importlib import resources
from typing import Any

from langchain_core.tools import BaseTool, tool


class CRM:
    def __init__(self, customers: list[dict[str, Any]], orders: list[dict[str, Any]]) -> None:
        self.customers = {c["id"]: c for c in customers}
        self.orders = {o["id"]: o for o in orders}

    @classmethod
    def bundled(cls) -> CRM:
        raw = json.loads(
            resources.files("support_agents.data").joinpath("crm.json").read_text("utf-8")
        )
        return cls(raw["customers"], raw["orders"])


def make_crm_tools(crm: CRM, today: Callable[[], date] = date.today) -> list[BaseTool]:
    @tool
    def get_customer(customer_id: str) -> str:
        """Look up a customer profile (name, country, membership, order ids) by customer id."""
        customer = crm.customers.get(customer_id.strip().upper())
        if customer is None:
            return f"Customer {customer_id} not found."
        return json.dumps(customer)

    @tool
    def get_order(order_id: str) -> str:
        """Look up an order (item, total, status, delivery date) by order id, e.g. ORD-1001."""
        order = crm.orders.get(order_id.strip().upper())
        if order is None:
            return f"Order {order_id} not found."
        result = dict(order)
        if order.get("delivered_on"):
            delivered = date.fromisoformat(order["delivered_on"])
            result["days_since_delivery"] = (today() - delivered).days
        return json.dumps(result)

    return [get_customer, get_order]
