from __future__ import annotations

import numpy as np

from qe.catalog.table import Column, Table
from qe.exec.compiler import compile_plan
from qe.plan.explain import explain
from qe.plan.logical import Limit as LimitNode
from qe.plan.optimizer import optimize
from qe.plan.planner import build_logical_plan
from qe.sql.ast import SelectQuery


def _materialize(op) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for batch in op.batches():
        for k, v in batch.items():
            chunks.setdefault(k, []).append(v)
    return {
        k: (np.concatenate(vs) if vs else np.array([], dtype=object))
        for k, vs in chunks.items()
    }


def _extract_limit_n(plan) -> int | None:
    # LIMIT is always the top-most node in your planner shape when present.
    if isinstance(plan, LimitNode):
        return plan.n
    return None


def _apply_limit(out_cols: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    for k in list(out_cols.keys()):
        out_cols[k] = out_cols[k][:n]
    return out_cols


def execute(
    query: SelectQuery,
    table: Table,
    batch_size: int = 4096,
    optimize_plan: bool = True,
    return_explain: bool = False,
):
    plan = build_logical_plan(query)
    plan_opt = optimize(plan) if optimize_plan else plan

    op = compile_plan(
        plan_opt, table, batch_size=batch_size, enable_limit_op=optimize_plan
    )
    out_cols = _materialize(op)

    if not optimize_plan:
        n = _extract_limit_n(plan_opt)
        if n is not None:
            out_cols = _apply_limit(out_cols, n)

    result = Table("result", {k: Column(k, "object", v) for k, v in out_cols.items()})

    if return_explain:
        return result, explain(plan), explain(plan_opt)

    return result
