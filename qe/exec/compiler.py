from __future__ import annotations

from qe.catalog.table import Table
from qe.exec.ops import AggregateOp, FilterOp, LimitOp, ProjectOp, ScanOp, SortOp
from qe.plan.logical import Aggregate, Filter, Limit, LogicalPlan, Project, Scan, Sort


def compile_plan(plan: LogicalPlan, table: Table, batch_size: int = 4096):
    if isinstance(plan, Scan):
        return ScanOp(table, batch_size=batch_size, needed_cols=plan.needed_cols)

    if isinstance(plan, Filter):
        return FilterOp(compile_plan(plan.child, table, batch_size), plan.predicate)

    if isinstance(plan, Project):
        return ProjectOp(
            compile_plan(plan.child, table, batch_size), plan.select, source_table=table
        )

    if isinstance(plan, Aggregate):
        return AggregateOp(
            compile_plan(plan.child, table, batch_size), plan.query, source_table=table
        )

    if isinstance(plan, Sort):
        ob = plan.order_by
        key = ob.expr.name
        return SortOp(
            compile_plan(plan.child, table, batch_size), key=key, ascending=ob.ascending
        )

    if isinstance(plan, Limit):
        return LimitOp(compile_plan(plan.child, table, batch_size), plan.n)

    raise TypeError(f"Unknown plan node: {type(plan)}")
