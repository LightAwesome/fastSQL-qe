from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence

import numpy as np

from qe.catalog.table import Table
from qe.errors import QueryError
from qe.exec.expr_eval import eval_expr
from qe.exec.naming import expr_to_name
from qe.sql.ast import AggFunc, ColRef, SelectItem

Batch = dict[str, np.ndarray]


class Op:
    def batches(self) -> Iterator[Batch]:
        raise NotImplementedError


class ScanOp(Op):
    def __init__(self, table: Table, batch_size: int = 4096, needed_cols: Optional[Sequence[str]] = None):
        self.table = table
        self.batch_size = batch_size
        self.needed = list(needed_cols) if needed_cols is not None else None

        if self.needed is not None:
            schema = set(self.table.col_names())
            missing = [c for c in self.needed if c not in schema]
            if missing:
                raise QueryError(f"Scan needed_cols contains unknown columns: {missing}")

    def batches(self) -> Iterator[Batch]:
        n = self.table.row_count
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            out: Batch = {}
            for name, col in self.table.columns.items():  # preserves schema order
                if self.needed is None or name in set(self.needed):
                    out[name] = col.data[start:end]
            yield out


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
        self.source_table = source_table

        # Expand SELECT *
        self.items: list[SelectItem] = []
        for item in select_items:
            if isinstance(item.expr, ColRef) and item.expr.name == "*":
                for col_name in self.source_table.col_names():
                    self.items.append(SelectItem(expr=ColRef(col_name), alias=None))
            else:
                self.items.append(item)

    def _output_name(self, item: SelectItem) -> str:
        alias = getattr(item, "alias", None)
        return alias or expr_to_name(item.expr)

    def batches(self) -> Iterator[Batch]:
        for batch in self.child.batches():
            n = len(next(iter(batch.values()))) if batch else 0
            out: Batch = {}
            for item in self.items:
                name = self._output_name(item)
                val = eval_expr(item.expr, batch)
                if isinstance(val, np.ndarray):
                    out[name] = val
                else:
                    out[name] = np.full(n, val, dtype=object)
            yield out


class LimitOp(Op):
    """Streaming LIMIT: short-circuits the pipeline once N rows have been produced."""
    def __init__(self, child: Op, n: int):
        if n < 0:
            raise QueryError("LIMIT must be non-negative")
        self.child = child
        self.remaining = n

    def batches(self) -> Iterator[Batch]:
        if self.remaining <= 0:
            return
        for batch in self.child.batches():
            if self.remaining <= 0:
                break
            if not batch:
                yield batch
                continue

            m = len(next(iter(batch.values())))
            if m <= self.remaining:
                self.remaining -= m
                yield batch
            else:
                out = {k: v[: self.remaining] for k, v in batch.items()}
                self.remaining = 0
                yield out
                break


class SortOp(Op):
    """
    Blocking ORDER BY (single key): materializes child, sorts once, yields one batch.
    """
    def __init__(self, child: Op, key: str, ascending: bool = True):
        self.child = child
        self.key = key
        self.ascending = ascending

    def batches(self) -> Iterator[Batch]:
        chunks: dict[str, list[np.ndarray]] = {}
        for batch in self.child.batches():
            for k, v in batch.items():
                chunks.setdefault(k, []).append(v)

        if not chunks:
            yield {}
            return

        cols = {k: np.concatenate(vs) if vs else np.array([], dtype=object) for k, vs in chunks.items()}
        if self.key not in cols:
            raise QueryError(f"ORDER BY column '{self.key}' must be selected")

        idx = np.argsort(cols[self.key])
        if not self.ascending:
            idx = idx[::-1]
        for k in list(cols.keys()):
            cols[k] = cols[k][idx]

        yield cols


class AggregateOp(Op):
    """
    Blocking hash aggregation: consumes all input and yields exactly one output batch.
    Deterministic group order: first-seen order (dict insertion order).
    """

    def __init__(self, child: Op, query, source_table: Table):
        self.child = child
        self.query = query
        self.source_table = source_table

        self.group_cols: list[str] = [g.name for g in (query.group_by or [])]
        self.select_items: list[SelectItem] = list(query.select)

        self.agg_specs: list[tuple[str, str, Any]] = []
        self.select_kinds: list[tuple[str, Any]] = []  # ("group", colname) or ("agg", spec_index)

        for item in self.select_items:
            alias = getattr(item, "alias", None)
            if isinstance(item.expr, ColRef):
                out_name = alias or expr_to_name(item.expr)
                self.select_kinds.append(("group", item.expr.name))
            elif isinstance(item.expr, AggFunc):
                out_name = alias or expr_to_name(item.expr)
                func = item.expr.func.upper()
                arg = item.expr.arg

                if isinstance(arg, ColRef) and arg.name == "*":
                    if func != "COUNT":
                        raise QueryError(f"{func}(*) not supported (only COUNT(*) allowed)")
                    self.agg_specs.append((out_name, func, None))
                else:
                    self.agg_specs.append((out_name, func, arg))

                self.select_kinds.append(("agg", len(self.agg_specs) - 1))
            else:
                raise QueryError("Only column references and aggregates are supported in SELECT with aggregation")

        src_cols = set(source_table.col_names())
        for c in self.group_cols:
            if c not in src_cols:
                raise QueryError(f"Unknown column in GROUP BY: '{c}'")

    def batches(self) -> Iterator[Batch]:
        groups: dict[tuple, dict[str, Any]] = {}

        def new_agg_states():
            states = []
            for (_out, func, _arg) in self.agg_specs:
                if func == "COUNT":
                    states.append({"count": 0})
                elif func == "SUM":
                    states.append({"sum": 0.0})
                elif func == "MIN":
                    states.append({"min": None})
                elif func == "MAX":
                    states.append({"max": None})
                elif func == "AVG":
                    states.append({"sum": 0.0, "count": 0})
                else:
                    raise QueryError(f"Unsupported aggregate: {func}")
            return states

        for batch in self.child.batches():
            n = len(next(iter(batch.values()))) if batch else 0
            if n == 0:
                continue

            group_arrays = [batch[c] for c in self.group_cols]

            agg_arg_arrays: list[np.ndarray | None] = []
            for (_out, func, arg) in self.agg_specs:
                if func == "COUNT" and arg is None:
                    agg_arg_arrays.append(None)
                else:
                    v = eval_expr(arg, batch)
                    if not isinstance(v, np.ndarray):
                        v = np.full(n, v, dtype=object)
                    agg_arg_arrays.append(v)

            for i in range(n):
                if self.group_cols:
                    key = tuple(arr[i].item() if hasattr(arr[i], "item") else arr[i] for arr in group_arrays)
                    group_vals = key
                else:
                    key = tuple()
                    group_vals = tuple()

                if key not in groups:
                    groups[key] = {"group_vals": group_vals, "aggs": new_agg_states()}

                states = groups[key]["aggs"]

                for si, (_out, func, _arg) in enumerate(self.agg_specs):
                    st = states[si]
                    if func == "COUNT":
                        st["count"] += 1
                    elif func == "SUM":
                        st["sum"] += float(agg_arg_arrays[si][i])
                    elif func == "MIN":
                        v = agg_arg_arrays[si][i]
                        st["min"] = v if st["min"] is None or v < st["min"] else st["min"]
                    elif func == "MAX":
                        v = agg_arg_arrays[si][i]
                        st["max"] = v if st["max"] is None or v > st["max"] else st["max"]
                    elif func == "AVG":
                        st["sum"] += float(agg_arg_arrays[si][i])
                        st["count"] += 1

        # Emit exactly one batch
        if not groups:
            yield {}
            return

        out_cols: dict[str, list[Any]] = {}
        for kind, payload in self.select_kinds:
            if kind == "group":
                out_cols[payload] = []
            else:
                spec_idx = payload
                out_name, _func, _arg = self.agg_specs[spec_idx]
                out_cols[out_name] = []

        for _key, state in groups.items():
            group_vals = state["group_vals"]
            agg_states = state["aggs"]

            for kind, payload in self.select_kinds:
                if kind == "group":
                    col = payload
                    j = self.group_cols.index(col)
                    out_cols[col].append(group_vals[j])
                else:
                    spec_idx = payload
                    out_name, func, _arg = self.agg_specs[spec_idx]
                    st = agg_states[spec_idx]
                    if func == "COUNT":
                        out_cols[out_name].append(st["count"])
                    elif func == "SUM":
                        out_cols[out_name].append(st["sum"])
                    elif func == "MIN":
                        out_cols[out_name].append(st["min"])
                    elif func == "MAX":
                        out_cols[out_name].append(st["max"])
                    elif func == "AVG":
                        out_cols[out_name].append(st["sum"] / st["count"] if st["count"] else 0.0)

        yield {name: np.array(vals, dtype=object) for name, vals in out_cols.items()}
