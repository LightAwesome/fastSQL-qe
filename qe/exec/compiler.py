from __future__ import annotations

from qe.catalog.table import Table
from qe.exec.ops import (
    AggregateOp,
    FilterOp,
    LimitOp,
    ProjectOp,
    ScanOp,
    SortOp,
    TopNOp,
)
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


def compile_plan(
    plan: LogicalPlan,
    table: Table,
    batch_size: int = 4096,
    enable_limit_op: bool = True,  # NEW
):
    if isinstance(plan, Scan):
        return ScanOp(table, batch_size=batch_size, needed_cols=plan.needed_cols)

    if isinstance(plan, Filter):
        return FilterOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op), plan.predicate
        )

    if isinstance(plan, Project):
        return ProjectOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op),
            plan.select,
            source_table=table,
        )

    if isinstance(plan, Aggregate):
        return AggregateOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op),
            plan.query,
            source_table=table,
        )

    if isinstance(plan, Sort):
        ob = plan.order_by
        key = ob.expr.name
        return SortOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op),
            key=key,
            ascending=ob.ascending,
        )

    if isinstance(plan, TopN):
        ob = plan.order_by
        key = ob.expr.name
        return TopNOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op),
            key=key,
            ascending=ob.ascending,
            n=plan.n,
        )

    if isinstance(plan, Limit):
        if not enable_limit_op:
            return compile_plan(plan.child, table, batch_size, enable_limit_op)
        return LimitOp(
            compile_plan(plan.child, table, batch_size, enable_limit_op), plan.n
        )

    raise TypeError(f"Unknown plan node: {type(plan)}")
