from __future__ import annotations

import numpy as np

from qe.catalog.table import Column, Table
from qe.errors import QueryError
from qe.exec.ops import AggregateOp, FilterOp, ProjectOp, ScanOp
from qe.sql.ast import AggFunc, ColRef, SelectQuery


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=object)
    return np.concatenate(chunks, axis=0)


def _materialize(op) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    for batch in op.batches():
        for name, arr in batch.items():
            collected.setdefault(name, []).append(arr)

    out: dict[str, np.ndarray] = {}
    for name in collected.keys():
        out[name] = _concat(collected[name])
    return out


def execute(query: SelectQuery, table: Table, batch_size: int = 4096) -> Table:
    op = ScanOp(table, batch_size=batch_size)

    if query.where is not None:
        from qe.exec.ops import FilterOp

        op = FilterOp(op, query.where)

    has_agg = any(isinstance(item.expr, AggFunc) for item in query.select)

    if query.group_by or has_agg:
        op = AggregateOp(op, query, source_table=table)
        out_cols = _materialize(op)
    else:
        op = ProjectOp(op, query.select, source_table=table)
        out_cols = _materialize(op)

    if query.order_by:
        if len(query.order_by) != 1:
            raise QueryError("Only single-key ORDER BY is supported")
        ob = query.order_by[0]
        if not isinstance(ob.expr, ColRef):
            raise QueryError("ORDER BY only supports column references")
        key = ob.expr.name
        if key not in out_cols:
            raise QueryError(f"ORDER BY column '{key}' must be selected")
        idx = np.argsort(out_cols[key])
        if not ob.ascending:
            idx = idx[::-1]
        for name in list(out_cols.keys()):
            out_cols[name] = out_cols[name][idx]

    if query.limit is not None:
        if query.limit < 0:
            raise QueryError("LIMIT must be non-negative")
        for name in list(out_cols.keys()):
            out_cols[name] = out_cols[name][: query.limit]

    columns = {name: Column(name, "object", arr) for name, arr in out_cols.items()}
    return Table("result", columns)
