from __future__ import annotations

from typing import Any

import numpy as np

from qe.catalog.table import Column, Table
from qe.errors import QueryError
from qe.exec.ops import FilterOp, ProjectOp, ScanOp
from qe.sql.ast import ColRef, SelectQuery


def _concat_batches(col_batches: list[np.ndarray]) -> np.ndarray:
    if not col_batches:
        return np.array([], dtype=object)
    return np.concatenate(col_batches, axis=0)


def execute(query: SelectQuery, table: Table, batch_size: int = 4096) -> Table:
    if query.group_by:
        raise QueryError("GROUP BY is not supported yet (Commit 8)")

    for item in query.select:
        pass

    op = ScanOp(table, batch_size=batch_size)

    if query.where is not None:
        op = FilterOp(op, query.where)

    op = ProjectOp(op, query.select, source_table=table)

    collected: dict[str, list[np.ndarray]] = {}
    for batch in op.batches():
        for name, arr in batch.items():
            collected.setdefault(name, []).append(arr)

    out_cols: dict[str, np.ndarray] = {}
    for name in collected.keys():
        out_cols[name] = _concat_batches(collected[name])

    if query.order_by:
        if len(query.order_by) != 1:
            raise QueryError("Only single-key ORDER BY is supported")
        ob = query.order_by[0]
        expr = ob.expr
        if not isinstance(expr, ColRef):
            raise QueryError("ORDER BY only supports column references in Commit 7")

        key = expr.name
        if key not in out_cols:
            # SQL allows ordering by non-selected columns, but we keep it simple for now.
            raise QueryError(f"ORDER BY column '{key}' must be selected in Commit 7")

        idx = np.argsort(out_cols[key])
        if not ob.ascending:
            idx = idx[::-1]

        for name in list(out_cols.keys()):
            out_cols[name] = out_cols[name][idx]

    if query.limit is not None:
        n = query.limit
        if n < 0:
            raise QueryError("LIMIT must be non-negative")
        for name in list(out_cols.keys()):
            out_cols[name] = out_cols[name][:n]

    columns = {}
    for name, arr in out_cols.items():
        columns[name] = Column(name, "object", arr)

    return Table("result", columns)
