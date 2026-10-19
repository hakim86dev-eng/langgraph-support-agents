"""A safe arithmetic evaluator exposed as a LangChain tool (no ``eval``)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
MAX_EXPRESSION_LENGTH = 200


def safe_eval(expression: str) -> float:
    """Evaluate +, -, *, /, %, ** and parentheses over numeric literals only."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression too long")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return _eval(ast.parse(expression.replace("^", "**"), mode="eval"))


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '129 * 0.9' or '(349 - 20) / 2'."""
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"
    return f"{expression} = {round(result, 4):g}"
