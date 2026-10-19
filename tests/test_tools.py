from __future__ import annotations

import json

import pytest

from support_agents.tools import CRM, KnowledgeBase, Toolkit, calculate, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("1 + 2 * 3", 7.0), ("(349 - 20) / 2", 164.5), ("2 ^ 3", 8.0), ("-4 + 10 % 3", -3.0)],
)
def test_safe_eval(expr: str, expected: float) -> None:
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "a + 1", "[1, 2]", "2 ** 100", "1" * 201, "len('x')"]
)
def test_safe_eval_rejects_unsafe_input(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_calculate_tool_formats_and_reports_errors() -> None:
    assert calculate.invoke({"expression": "129 * 0.9"}) == "129 * 0.9 = 116.1"
    assert calculate.invoke({"expression": "1 / 0"}).startswith("error:")
    assert calculate.invoke({"expression": "os.system('x')"}).startswith("error:")


def test_knowledge_base_search_ranks_relevant_article() -> None:
    kb = KnowledgeBase.bundled()
    assert kb.search("how long does express shipping take")[0].id == "KB-102"
    assert kb.search("warranty for water damage")[0].id == "KB-103"
    assert kb.search("is my order cancellable")[0].id == "KB-104"
    assert kb.search("zzzz qqqq") == []


def test_kb_tool_output(toolkit: Toolkit) -> None:
    (search,) = toolkit["knowledge"]
    assert "KB-101" in search.invoke({"query": "return refund"})
    assert search.invoke({"query": "zzzz"}) == "No matching help-centre articles."


def test_crm_tools(toolkit: Toolkit) -> None:
    get_customer, get_order = toolkit["account"]
    customer = json.loads(get_customer.invoke({"customer_id": "cust-1"}))
    assert customer["name"] == "Aisha Khan"
    order = json.loads(get_order.invoke({"order_id": "ORD-1001"}))
    assert order["days_since_delivery"] == 18
    shipped = json.loads(get_order.invoke({"order_id": "ORD-1002"}))
    assert "days_since_delivery" not in shipped
    assert get_order.invoke({"order_id": "ORD-9"}) == "Order ORD-9 not found."
    assert get_customer.invoke({"customer_id": "X"}) == "Customer X not found."


def test_bundled_crm_is_consistent() -> None:
    crm = CRM.bundled()
    for customer in crm.customers.values():
        for order_id in customer["orders"]:
            assert crm.orders[order_id]["customer_id"] == customer["id"]
