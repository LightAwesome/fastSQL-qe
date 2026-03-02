from __future__ import annotations

import numpy as np

from qe.catalog.table import Column, Table
from qe.exec.compiler import compile_plan
from qe.plan.explain import explain
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


def execute(
    query: SelectQuery,
    table: Table,
    batch_size: int = 4096,
    optimize_plan: bool = True,
    return_explain: bool = False,
):
    plan = build_logical_plan(query)
    plan_opt = optimize(plan) if optimize_plan else plan

    op = compile_plan(plan_opt, table, batch_size=batch_size)
    out_cols = _materialize(op)

    result = Table("result", {k: Column(k, "object", v) for k, v in out_cols.items()})

    if return_explain:
        return result, explain(plan), explain(plan_opt)

    return result
