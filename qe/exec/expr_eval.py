import numpy as np

from ..catalog.table import Table
from ..errors import QueryError
from ..sql.ast import *


def eval_expr(expr, columns: dict) -> np.ndarray:
    """
    Evaluate expression over a batch of columns.
    columns: dict of col_name -> np.ndarray
    Returns np.ndarray of results.
    """
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, ColRef):
        if expr.name not in columns:
            raise QueryError(f"Column '{expr.name}' not found in batch")
        return columns[expr.name]

    if isinstance(expr, UnaryOp):
        operand = eval_expr(expr.operand, columns)
        if expr.op == "-":
            return -operand
        if expr.op == "not":
            return ~operand.astype(bool)
        raise QueryError(f"Unknown unary op: {expr.op}")

    if isinstance(expr, BinOp):
        left = eval_expr(expr.left, columns)
        right = eval_expr(expr.right, columns)
        op = expr.op

        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if np.issubdtype(np.array(right).dtype, np.integer):
                if np.any(np.array(right) == 0):
                    raise QueryError("Division by zero")
            return left / right
        if op == "%":
            return left % right

        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right

        if op == "and":
            return left.astype(bool) & right.astype(bool)
        if op == "or":
            return left.astype(bool) | right.astype(bool)
        raise QueryError(f"Unknown binary op: {op!r}")

    raise QueryError(f"Cannot evaluate expression type: {type(expr).__name__}")
