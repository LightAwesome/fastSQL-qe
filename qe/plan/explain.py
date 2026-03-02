from __future__ import annotations

from qe.plan.logical import (
    Aggregate,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Scan,
    Sort,
    TopN,
)
from qe.sql.ast import AggFunc, BinOp, ColRef, Literal, UnaryOp


def explain(plan: LogicalPlan) -> str:
    lines: list[str] = []
    _walk(plan, lines, indent=0)
    return "\n".join(lines)


def _walk(plan: LogicalPlan, lines: list[str], indent: int) -> None:
    pad = "  " * indent

    if isinstance(plan, Scan):
        cols = "*" if plan.needed_cols is None else ", ".join(plan.needed_cols)
        lines.append(f"{pad}Scan(table={plan.table_name}, cols=[{cols}])")
        return

    if isinstance(plan, Filter):
        lines.append(f"{pad}Filter(predicate={_expr_str(plan.predicate)})")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Project):
        names = ", ".join(_expr_str(item.expr) for item in plan.select)
        lines.append(f"{pad}Project(select=[{names}])")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Aggregate):
        gb = (
            []
            if plan.query.group_by is None
            else [_expr_str(g) for g in plan.query.group_by]
        )
        aggs = [_expr_str(item.expr) for item in plan.query.select]
        lines.append(
            f"{pad}Aggregate(group_by=[{', '.join(gb)}], select=[{', '.join(aggs)}])"
        )
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Sort):
        ob = plan.order_by
        direction = "ASC" if ob.ascending else "DESC"
        lines.append(f"{pad}Sort(key={_expr_str(ob.expr)}, direction={direction})")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Limit):
        lines.append(f"{pad}Limit(n={plan.n})")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, TopN):
        lines.append(f"{pad}TopN(n={plan.n})")
        _walk(plan.child, lines, indent + 1)
        return

    lines.append(f"{pad}{type(plan).__name__}")


def _expr_str(expr: object) -> str:
    """Render an AST expression as a readable string for EXPLAIN output."""
    if expr is None:
        return "None"
    if isinstance(expr, ColRef):
        return expr.name
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, UnaryOp):
        return f"({expr.op} {_expr_str(expr.operand)})"
    if isinstance(expr, BinOp):
        return f"({_expr_str(expr.left)} {expr.op} {_expr_str(expr.right)})"
    if isinstance(expr, AggFunc):
        return f"{expr.func}({_expr_str(expr.arg)})"
    return repr(expr)
