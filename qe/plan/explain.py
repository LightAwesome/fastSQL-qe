from __future__ import annotations

from qe.plan.logical import (Aggregate, Filter, Limit, LogicalPlan, Project,
                             Scan, Sort)


def explain(plan: LogicalPlan) -> str:
    lines: list[str] = []
    _walk(plan, lines, indent=0)
    return "\n".join(lines)

def _walk(plan: LogicalPlan, lines: list[str], indent: int) -> None:
    pad = "  " * indent

    if isinstance(plan, Scan):
        cols = "*" if plan.needed_cols is None else ",".join(plan.needed_cols)
        lines.append(f"{pad}Scan(table={plan.table_name}, cols={cols})")
        return

    if isinstance(plan, Filter):
        lines.append(f"{pad}Filter(predicate=...)")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Project):
        lines.append(f"{pad}Project(select={len(plan.select)})")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Aggregate):
        gb = 0 if plan.query.group_by is None else len(plan.query.group_by)
        lines.append(f"{pad}Aggregate(group_by={gb})")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Sort):
        lines.append(f"{pad}Sort(order_by=1)")
        _walk(plan.child, lines, indent + 1)
        return

    if isinstance(plan, Limit):
        lines.append(f"{pad}Limit(n={plan.n})")
        _walk(plan.child, lines, indent + 1)
        return

    lines.append(f"{pad}{type(plan).__name__}")
