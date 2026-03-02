from __future__ import annotations

from qe.errors import QueryError
from qe.plan.logical import Aggregate, Filter, Limit, Project, Scan, Sort
from qe.sql.ast import AggFunc, SelectQuery


def build_logical_plan(q: SelectQuery):
    plan = Scan(q.from_table)

    if q.where is not None:
        plan = Filter(plan, q.where)

    has_agg = any(isinstance(item.expr, AggFunc) for item in q.select)

    if q.group_by or has_agg:
        plan = Aggregate(plan, q)
    else:
        plan = Project(plan, q.select)

    if q.order_by:
        if len(q.order_by) != 1:
            raise QueryError("Only single-key ORDER BY is supported")
        plan = Sort(plan, q.order_by[0])

    if q.limit is not None:
        if q.limit < 0:
            raise QueryError("LIMIT must be non-negative")
        plan = Limit(plan, q.limit)

    return plan
