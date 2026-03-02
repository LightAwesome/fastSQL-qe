from __future__ import annotations

from typing import Any

import numpy as np

from qe.errors import QueryError
from qe.sql.ast import AggFunc, BinOp, ColRef, Literal, UnaryOp


def _batch_len(batch: dict[str, np.ndarray]) -> int:
    if not batch:
        return 0
    return len(next(iter(batch.values())))


def _ensure_array(val: Any, n: int) -> np.ndarray:
    """Broadcast scalar -> length-n array. Keep arrays as-is."""
    if isinstance(val, np.ndarray):
        return val
    return np.full(n, val, dtype=object)


def _ensure_bool_array(val: Any, n: int, ctx: str) -> np.ndarray:
    arr = _ensure_array(val, n)
    if not (isinstance(arr, np.ndarray) and arr.dtype == bool):
        raise QueryError(f"{ctx} must evaluate to a boolean array")
    return arr


def eval_expr(expr: Any, batch: dict[str, np.ndarray]) -> Any:
    """
    Evaluate expression against a batch (column-name -> numpy array slice).
    Returns: numpy array (most cases) or scalar (Literal).
    """
    n = _batch_len(batch)

    if isinstance(expr, ColRef):
        if expr.name == "*":
            raise QueryError("'*' is not a valid expression here")
        if expr.name not in batch:
            raise QueryError(f"Unknown column in expression: '{expr.name}'")
        return batch[expr.name]

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, UnaryOp):
        op = expr.op.lower()
        v = eval_expr(expr.operand, batch)

        if op == "not":
            return np.logical_not(_ensure_bool_array(v, n, "NOT operand"))
        if op == "-":
            return -_ensure_array(v, n)

        raise QueryError(f"Unsupported unary operator: {expr.op}")

    if isinstance(expr, BinOp):
        op = expr.op.lower()
        lhs = eval_expr(expr.left, batch)
        rhs = eval_expr(expr.right, batch)

        if op == "and":
            return np.logical_and(
                _ensure_bool_array(lhs, n, "AND lhs"),
                _ensure_bool_array(rhs, n, "AND rhs"),
            )
        if op == "or":
            return np.logical_or(
                _ensure_bool_array(lhs, n, "OR lhs"),
                _ensure_bool_array(rhs, n, "OR rhs"),
            )

        a = _ensure_array(lhs, n)
        b = _ensure_array(rhs, n)

        if op == "=":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b

        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "%":
            return a % b

        raise QueryError(f"Unsupported binary operator: {expr.op}")

    if isinstance(expr, AggFunc):
        raise QueryError("Aggregates are not supported in Commit 7 execution")

    raise QueryError(f"Unknown expression type in eval: {type(expr)}")
