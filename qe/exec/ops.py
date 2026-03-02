from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from qe.catalog.table import Table
from qe.errors import QueryError
from qe.exec.expr_eval import eval_expr
from qe.sql.ast import ColRef, SelectItem

Batch = dict[str, np.ndarray]


class Op:
    def batches(self) -> Iterator[Batch]:
        raise NotImplementedError


class ScanOp(Op):
    def __init__(self, table: Table, batch_size: int = 4096):
        self.table = table
        self.batch_size = batch_size

    def batches(self) -> Iterator[Batch]:
        cols = self.table.columns
        n = self.table.row_count
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield {name: col.data[start:end] for name, col in cols.items()}


class FilterOp(Op):
    def __init__(self, child: Op, predicate: Any):
        self.child = child
        self.predicate = predicate

    def batches(self) -> Iterator[Batch]:
        for batch in self.child.batches():
            mask = eval_expr(self.predicate, batch)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise QueryError("WHERE clause must evaluate to a boolean mask")
            yield {k: v[mask] for k, v in batch.items()}


class ProjectOp(Op):
    def __init__(self, child: Op, select_items: list[SelectItem], source_table: Table):
        self.child = child
        self.select_items = select_items
        self.source_table = source_table

        self._expanded_items: list[SelectItem] = []
        for item in self.select_items:
            if isinstance(item.expr, ColRef) and item.expr.name == "*":
                for col_name in self.source_table.col_names():
                    self._expanded_items.append(
                        SelectItem(expr=ColRef(col_name), alias=None)
                    )
            else:
                self._expanded_items.append(item)

    def _output_name(self, idx: int, item: SelectItem) -> str:
        if getattr(item, "alias", None):
            return item.alias
        if isinstance(item.expr, ColRef):
            return item.expr.name
        return f"expr_{idx}"

    def batches(self) -> Iterator[Batch]:
        for batch in self.child.batches():
            n = len(next(iter(batch.values()))) if batch else 0
            out: Batch = {}
            for i, item in enumerate(self._expanded_items):
                name = self._output_name(i, item)
                val = eval_expr(item.expr, batch)

                # Ensure column is an array of length n
                if isinstance(val, np.ndarray):
                    arr = val
                else:
                    arr = np.full(n, val, dtype=object)

                out[name] = arr
            yield out
