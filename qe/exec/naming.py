from __future__ import annotations

from qe.sql.ast import AggFunc, BinOp, ColRef, Literal, UnaryOp

_OP_WORD = {
    "+": "plus",
    "-": "minus",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "=": "eq",
    "!=": "neq",
    "<": "lt",
    "<=": "lte",
    ">": "gt",
    ">=": "gte",
    "and": "and",
    "or": "or",
}


def expr_to_name(expr: object) -> str:
    """
    Deterministic column names.
    Literals collapse to 'const' (stable across literal value changes).
    Alias (AS ...) should override this upstream.
    """
    if isinstance(expr, ColRef):
        return "star" if expr.name == "*" else expr.name

    if isinstance(expr, Literal):
        return "const"

    if isinstance(expr, UnaryOp):
        op = expr.op.lower()
        if op == "not":
            return f"not_{expr_to_name(expr.operand)}"
        if op == "-":
            return f"neg_{expr_to_name(expr.operand)}"
        return f"un_{op}_{expr_to_name(expr.operand)}"

    if isinstance(expr, BinOp):
        op = expr.op.lower()
        w = _OP_WORD.get(op, op)
        return f"{expr_to_name(expr.left)}_{w}_{expr_to_name(expr.right)}"

    if isinstance(expr, AggFunc):
        func = expr.func.lower()
        arg = expr.arg
        if isinstance(arg, ColRef) and arg.name == "*":
            return f"{func}_star"
        return f"{func}_{expr_to_name(arg)}"

    return "expr"
