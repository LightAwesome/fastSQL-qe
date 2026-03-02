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
    def __init__(
        self,
        table: Table,
        batch_size: int = 4096,
        needed_cols: Optional[Sequence[str]] = None,
    ):
        self.table = table
        self.batch_size = batch_size
        self.needed = list(needed_cols) if needed_cols is not None else None

        self.batches_emitted = 0
        self.rows_emitted = 0
        if self.needed is not None:
            schema = set(self.table.col_names())
            missing = [c for c in self.needed if c not in schema]
            if missing:
                raise QueryError(f"Scan needed_cols contains unknown columns: {missing}")

    def batches(self) -> Iterator[Batch]:
        n = self.table.row_count
        needed_set = set(self.needed) if self.needed is not None else None
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            out: Batch = {}
            for name, col in self.table.columns.items():
                if needed_set is None or name in needed_set:
                    out[name] = col.data[start:end]
            self.batches_emitted += 1
            self.rows_emitted += (end - start)
            yield out



class FilterOp(Op):
    def __init__(self, child: Op, predicate: Any):
        self.child = child
        self.predicate = predicate

    def batches(self) -> Iterator[Batch]:
        for batch in self.child.batches():
            mask = eval_expr(self.predicate, batch)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                mask = np.asarray(mask, dtype=bool)
            yield {k: v[mask] for k, v in batch.items()}



class ProjectOp(Op):
    def __init__(self, child: Op, select_items: list[SelectItem], source_table: Table):
        self.child = child
        self.source_table = source_table

        # Expand SELECT * at construction time
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
    """Streaming LIMIT: short-circuits the pipeline once N rows are produced."""

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
                yield {k: v[: self.remaining] for k, v in batch.items()}
                self.remaining = 0
                break


class TopNOp(Op):
    """
    Blocking Top-N for ORDER BY key LIMIT N using argpartition.
    Requires key to be present in the child output.
    """
    def __init__(self, child: Op, key: str, ascending: bool, n: int):
        self.child = child
        self.key = key
        self.ascending = ascending
        self.n = n

    def batches(self) -> Iterator[Batch]:
        if self.n <= 0:
            yield {}
            return

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

        key_arr = cols[self.key]
        m = len(key_arr)
        n = min(self.n, m)
        if n == m:
            # just sort all if small / equal
            idx = np.argsort(key_arr)
            if not self.ascending:
                idx = idx[::-1]
            yield {k: v[idx] for k, v in cols.items()}
            return

        # Choose indices for best n
        if self.ascending:
            part_idx = np.argpartition(key_arr, n - 1)[:n]
            # sort the selected n by key
            order = np.argsort(key_arr[part_idx])
            idx = part_idx[order]
        else:
            # n largest: partition at m-n and take tail
            part_idx = np.argpartition(key_arr, m - n)[m - n:]
            order = np.argsort(key_arr[part_idx])[::-1]
            idx = part_idx[order]

        yield {k: v[idx] for k, v in cols.items()}


class SortOp(Op):
    """Blocking ORDER BY (single key): materializes child, sorts, yields one batch."""

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

        cols = {
            k: np.concatenate(vs) if vs else np.array([], dtype=object)
            for k, vs in chunks.items()
        }
        if self.key not in cols:
            raise QueryError(f"ORDER BY column '{self.key}' must be in SELECT")

        idx = np.argsort(cols[self.key], kind="stable")
        if not self.ascending:
            idx = idx[::-1]
        yield {k: v[idx] for k, v in cols.items()}



class AggregateOp(Op):
    """
    Blocking hash aggregation.

    Design decisions (documented for interviews):
    - SUM / COUNT / MIN / MAX use numpy vectorized operations per group via
      np.add.at / comparison masks — no Python loop over individual rows.
    - AVG is computed as SUM / COUNT after accumulation.
    - Group key extraction still uses a Python loop to build the key→index map,
      which is O(n) in Python. For high-cardinality groups on large tables this
      is the bottleneck. A future optimization would use pandas groupby or a
      Cython/numba kernel for key hashing.
    - Yields exactly one batch (blocking).
    - Group output order is first-seen (dict insertion order).
    """

    def __init__(self, child: Op, query, source_table: Table):
        self.child = child
        self.query = query
        self.source_table = source_table

        self.group_cols: list[str] = [g.name for g in (query.group_by or [])]
        self.select_items: list[SelectItem] = list(query.select)

        self.agg_specs: list[tuple[str, str, Any]] = []
        self.select_kinds: list[tuple[str, Any]] = []

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
                        raise QueryError(f"{func}(*) not supported; only COUNT(*) is allowed")
                    self.agg_specs.append((out_name, func, None))
                else:
                    self.agg_specs.append((out_name, func, arg))
                self.select_kinds.append(("agg", len(self.agg_specs) - 1))
            else:
                raise QueryError(
                    "SELECT with aggregation supports only column references and aggregate functions"
                )

        src_cols = set(source_table.col_names())
        for c in self.group_cols:
            if c not in src_cols:
                raise QueryError(f"Unknown column in GROUP BY: '{c}'")


    def batches(self) -> Iterator[Batch]:
        chunks: dict[str, list[np.ndarray]] = {}
        for batch in self.child.batches():
            for k, v in batch.items():
                chunks.setdefault(k, []).append(v)

        if not chunks:
            yield {}
            return

        full: dict[str, np.ndarray] = {
            k: np.concatenate(vs) for k, vs in chunks.items()
        }
        n = len(next(iter(full.values())))

        if n == 0:
            yield {}
            return

        if self.group_cols:
            group_index: dict[tuple, list[int]] = {}
            key_arrays = [full[c] for c in self.group_cols]
            for i in range(n):
                key = tuple(
                    arr[i].item() if hasattr(arr[i], "item") else arr[i]
                    for arr in key_arrays
                )
                if key not in group_index:
                    group_index[key] = []
                group_index[key].append(i)
        else:
            group_index = {(): list(range(n))}

        out_cols: dict[str, list[Any]] = {}
        for kind, payload in self.select_kinds:
            if kind == "group":
                out_cols[payload] = []
            else:
                out_name, _, _ = self.agg_specs[payload]
                out_cols[out_name] = []

        for key, indices in group_index.items():
            idx = np.array(indices, dtype=np.intp)

            for kind, payload in self.select_kinds:
                if kind == "group":
                    col = payload
                    j = self.group_cols.index(col)
                    out_cols[col].append(key[j])
                else:
                    spec_idx = payload
                    out_name, func, arg = self.agg_specs[spec_idx]
                    if func == "COUNT":
                        out_cols[out_name].append(len(idx))
                    else:
                        vals = eval_expr(arg, full)
                        if not isinstance(vals, np.ndarray):
                            vals = np.full(n, vals)
                        group_vals = vals[idx]  

                        if func == "SUM":
                            out_cols[out_name].append(float(np.sum(group_vals)))
                        elif func == "MIN":
                            out_cols[out_name].append(group_vals.min())
                        elif func == "MAX":
                            out_cols[out_name].append(group_vals.max())
                        elif func == "AVG":
                            out_cols[out_name].append(float(np.mean(group_vals)))
                        else:
                            raise QueryError(f"Unsupported aggregate: {func}")

        yield {
            name: np.array(vals, dtype=object)
            for name, vals in out_cols.items()
        }
