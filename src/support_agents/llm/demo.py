"""Offline, rule-based responder so the service works end-to-end without an API key.

It is intentionally simple (keyword routing and templated replies). It is *not* meant to be
smart; it exercises the real graph, tools, checkpointing and API exactly as an LLM would.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from support_agents.llm.fake import ToolSpec, role_of, tool_call_message

_ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_CUSTOMER_RE = re.compile(r"\bCUST-\d+\b", re.IGNORECASE)
_EXPR_RE = re.compile(r"[\d.]+(?:\s*[-+*/x]\s*\(?[\d.]+\)?)+")
_FINDING_RE = re.compile(r"^\[(\d+)\] \((\w+)\) (.+)$", re.MULTILINE)
_POLICY_WORDS = (
    "return", "refund", "ship", "deliver", "warranty", "cancel", "plus", "membership",
    "pay", "instalment", "installment", "cost", "price", "policy", "defect", "broken",
)  # fmt: skip
_ACCOUNT_WORDS = ("my account", "my order", "my orders", "membership", "am i", "my profile")
_SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\W*$", re.I)


def _field(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _last_human(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _plan(text: str) -> AIMessage:
    request = _field(text, "Customer request")
    customer_id = _field(text, "Customer id")
    lowered = request.lower()
    steps: list[dict[str, str]] = []
    if _SMALL_TALK.match(request):
        return tool_call_message("Plan", {"steps": [], "rationale": "small talk"})
    for order_id in dict.fromkeys(o.upper() for o in _ORDER_RE.findall(request)):
        steps.append({"worker": "account", "instruction": f"Look up order {order_id}"})
    if not steps and customer_id != "unknown" and any(w in lowered for w in _ACCOUNT_WORDS):
        steps.append({"worker": "account", "instruction": f"Look up customer {customer_id}"})
    if any(w in lowered for w in _POLICY_WORDS) or not steps:
        steps.append({"worker": "knowledge", "instruction": request})
    if expr := _EXPR_RE.search(request):
        steps.append({"worker": "math", "instruction": f"Compute: {expr.group(0)}"})
    return tool_call_message("Plan", {"steps": steps, "rationale": "keyword routing (demo)"})


def _summarise_tool_output(tool: str, output: str) -> str:
    if tool == "get_order" and output.startswith("{"):
        o = json.loads(output)
        text = f"Order {o['id']} ({o['item']}, {o['total_usd']} USD) is {o['status']}"
        if o.get("delivered_on"):
            text += f"; delivered on {o['delivered_on']} ({o['days_since_delivery']} days ago)"
        return text + "."
    if tool == "get_customer" and output.startswith("{"):
        c = json.loads(output)
        return (
            f"{c['name']} is a {c['membership']} member in {c['country']} "
            f"with orders {', '.join(c['orders'])}."
        )
    if tool == "search_knowledge_base" and " - " in output:
        article = output.split("\n\n")[0]
        return article.split(": ", 1)[-1]
    return output[:400]


def _worker(messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage:
    last = messages[-1]
    if isinstance(last, ToolMessage):
        previous = messages[-2]
        tool = previous.tool_calls[0]["name"] if isinstance(previous, AIMessage) else ""
        return AIMessage(content=_summarise_tool_output(tool, str(last.content)))
    names = {t["function"]["name"] for t in tools}
    instruction = _field(_last_human(messages), "Your instruction")
    if "search_knowledge_base" in names:
        return tool_call_message("search_knowledge_base", {"query": instruction})
    if "get_order" in names and (order := _ORDER_RE.search(instruction)):
        return tool_call_message("get_order", {"order_id": order.group(0)})
    if "get_customer" in names and (customer := _CUSTOMER_RE.search(instruction)):
        return tool_call_message("get_customer", {"customer_id": customer.group(0)})
    if "calculate" in names and (expr := _EXPR_RE.search(instruction)):
        return tool_call_message("calculate", {"expression": expr.group(0).replace("x", "*")})
    return AIMessage(content="I could not complete this step with my tools.")


def _write(text: str) -> AIMessage:
    findings = _FINDING_RE.findall(text)
    if not findings:
        return AIMessage(
            content="Hi! Thanks for reaching out to Orbit Electronics. Could you share a few "
            "more details, such as your order number, so I can help?"
        )
    body = " ".join(f"{summary} [{n}]" for n, _, summary in findings)
    return AIMessage(
        content=f"Thanks for contacting Orbit Electronics. {body} "
        "Let me know if there is anything else I can help with."
    )


def _review(text: str) -> AIMessage:
    numbers = {n for n, _, _ in _FINDING_RE.findall(text.split("DRAFT REPLY:")[0])}
    draft = text.split("DRAFT REPLY:", 1)[-1]
    missing = sorted(n for n in numbers if f"[{n}]" not in draft)
    if missing:
        feedback = f"Cite findings {', '.join(missing)}."
        return tool_call_message("Review", {"approved": False, "score": 2, "feedback": feedback})
    return tool_call_message("Review", {"approved": True, "score": 5, "feedback": ""})


def demo_responder(messages: Sequence[BaseMessage], tools: Sequence[ToolSpec]) -> AIMessage:
    role = role_of(messages)
    text = _last_human(messages)
    if role == "planner":
        return _plan(text)
    if role.startswith("worker:"):
        return _worker(messages, tools)
    if role == "writer":
        return _write(text)
    if role == "reviewer":
        return _review(text)
    return AIMessage(content="")
