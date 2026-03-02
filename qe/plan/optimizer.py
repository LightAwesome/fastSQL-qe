from __future__ import annotations

from qe.plan.logical import Aggregate, Filter, Limit, LogicalPlan, Project, Scan, Sort
from qe.sql.ast import AggFunc, BinOp, ColRef, UnaryOp


def optimize(plan: LogicalPlan) -> LogicalPlan:
    """
    Minimal optimizer:
      - Projection pushdown: compute required base-table columns and annotate Scan(needed_cols=...).
    """
    needed = _required_columns(plan)
    if not needed:
        return plan
    return _annotate_scan(plan, needed)


def _annotate_scan(plan: LogicalPlan, needed: set[str]) -> LogicalPlan:
    if isinstance(plan, Scan):
        return Scan(plan.table_name, needed_cols=sorted(needed))
    if isinstance(plan, Filter):
        return Filter(_annotate_scan(plan.child, needed), plan.predicate)
    if isinstance(plan, Project):
        return Project(_annotate_scan(plan.child, needed), plan.select)
    if isinstance(plan, Aggregate):
        return Aggregate(_annotate_scan(plan.child, needed), plan.query)
    if isinstance(plan, Sort):
        return Sort(_annotate_scan(plan.child, needed), plan.order_by)
    if isinstance(plan, Limit):
        return Limit(_annotate_scan(plan.child, needed), plan.n)
    return plan


def _required_columns(plan: LogicalPlan) -> set[str]:
    if isinstance(plan, Scan):
        return set()

    if isinstance(plan, Filter):
        return _required_columns(plan.child) | _colrefs(plan.predicate)

    if isinstance(plan, Project):
        base = _required_columns(plan.child)
        need = set()
        for item in plan.select:
            if isinstance(item.expr, ColRef) and item.expr.name == "*":
                return set()
            need |= _colrefs(item.expr)
        return base | need

    if isinstance(plan, Aggregate):
        base = _required_columns(plan.child)
        q = plan.query
        need = set()
        if q.group_by:
            for g in q.group_by:
                if isinstance(g, ColRef) and g.name != "*":
                    need.add(g.name)
        for item in q.select:
            if isinstance(item.expr, AggFunc):
                arg = item.expr.arg
                if isinstance(arg, ColRef) and arg.name == "*":
                    continue
                need |= _colrefs(arg)
            else:
                need |= _colrefs(item.expr)
        return base | need

    if isinstance(plan, Sort):
        return _required_columns(plan.child) | _colrefs(plan.order_by.expr)

    if isinstance(plan, Limit):
        return _required_columns(plan.child)

    return set()


def _colrefs(expr: object) -> set[str]:
    if expr is None:
        return set()
    if isinstance(expr, ColRef):
        return set() if expr.name == "*" else {expr.name}
    if isinstance(expr, BinOp):
        return _colrefs(expr.left) | _colrefs(expr.right)
    if isinstance(expr, UnaryOp):
        return _colrefs(expr.operand)
    if isinstance(expr, AggFunc):
        arg = expr.arg
        if isinstance(arg, ColRef) and arg.name == "*":
            return set()
        return _colrefs(arg)
    return set()
