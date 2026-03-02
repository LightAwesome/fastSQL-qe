from __future__ import annotations

import time

import numpy as np

from qe.catalog.table import Column, Table
from qe.exec.compiler import compile_plan
from qe.exec.ops import ScanOp
from qe.plan.analyzer import analyze
from qe.plan.explain import explain
from qe.plan.optimizer import optimize
from qe.plan.planner import build_logical_plan
from qe.sql.parser import Parser


def make_table(n: int) -> Table:
    rng = np.random.default_rng(0)
    categories = np.where(rng.random(n) > 0.5, "A", "B").astype(object)

    cols = {
        "id": Column("id", "int64", np.arange(n, dtype=np.int64)),
        "value": Column("value", "float64", rng.random(n)),
        "active": Column("active", "bool", rng.random(n) > 0.5),
        "category": Column("category", "object", categories),
    }

    # Wide table to make projection pushdown matter
    for i in range(32):
        cols[f"extra_{i}"] = Column(f"extra_{i}", "float64", rng.random(n))

    return Table("t", cols)


def materialize(op) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for batch in op.batches():
        for k, v in batch.items():
            chunks.setdefault(k, []).append(v)
    return {
        k: (np.concatenate(vs) if vs else np.array([], dtype=object))
        for k, vs in chunks.items()
    }


def find_scan(op) -> ScanOp | None:
    # Walk down common unary operator chain to locate the ScanOp for counters
    cur = op
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ScanOp):
            return cur
        nxt = getattr(cur, "child", None)
        cur = nxt
    return None


def run_once(
    sql: str, t: Table, optimize_plan: bool
) -> tuple[float, int, int, str, str]:
    q = Parser(sql).parse()
    analyze(q, t)

    plan = build_logical_plan(q)
    plan_opt = optimize(plan) if optimize_plan else plan

    # Make "no-opt" naive for LIMIT: disable LimitOp only in baseline
    enable_limit_op = optimize_plan

    op = compile_plan(plan_opt, t, batch_size=4096, enable_limit_op=enable_limit_op)
    scan = find_scan(op)

    t0 = time.perf_counter_ns()
    out = materialize(op)

    # If baseline disabled LimitOp, apply LIMIT after materialize (naive)
    if not optimize_plan and getattr(q, "limit", None) is not None:
        n = q.limit
        for k in list(out.keys()):
            out[k] = out[k][:n]

    t1 = time.perf_counter_ns()
    ms = (t1 - t0) / 1e6

    rows = scan.rows_emitted if scan else -1
    batches = scan.batches_emitted if scan else -1

    return ms, rows, batches, explain(plan), explain(plan_opt)


def summarize(times_ms: list[float]) -> tuple[float, float]:
    arr = np.array(times_ms, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def main():
    n = 10_000_000
    print(f"Generating {n:,} rows ... ", end="", flush=True)
    t = make_table(n)
    print("done\n")

    queries = [
        ("LIMIT early stop", "SELECT id FROM t LIMIT 10"),
        ("Selective filter + projection", "SELECT id FROM t WHERE value > 0.999"),
        (
            "Compute expression",
            "SELECT value * 2 FROM t WHERE active = true LIMIT 1000",
        ),
        (
            "Filter + GROUP BY (optimizer impact)",
            "SELECT active, SUM(value) FROM t WHERE value > 0.999 GROUP BY active",
        ),
        (
            "GROUP BY low cardinality (no filter)",
            "SELECT category, SUM(value) FROM t GROUP BY category",
        ),
        ("Sort + LIMIT", "SELECT id, value FROM t ORDER BY value LIMIT 100"),
    ]

    WARMUP = 1
    TRIALS = 5

    print(
        f"{'Query':45}  {'no-opt med/p95 (ms)':18}  {'opt med/p95 (ms)':18}  {'speedup':8}  {'opt rows/batches':14}"
    )
    print("-" * 115)

    explain_blocks = []

    for label, sql in queries:
        # Warmup
        for _ in range(WARMUP):
            run_once(sql, t, optimize_plan=True)

        no_times = []
        for _ in range(TRIALS):
            ms, _, _, before, after = run_once(sql, t, optimize_plan=False)
            no_times.append(ms)

        opt_times = []
        opt_rows = opt_batches = None
        for _ in range(TRIALS):
            ms, rows, batches, before, after = run_once(sql, t, optimize_plan=True)
            opt_times.append(ms)
            opt_rows, opt_batches = rows, batches

        no_med, no_p95 = summarize(no_times)
        opt_med, opt_p95 = summarize(opt_times)
        speed = (no_med / opt_med) if opt_med > 0 else float("inf")

        print(
            f"{label:45}  {no_med:7.1f}/{no_p95:7.1f}        {opt_med:7.1f}/{opt_p95:7.1f}        {speed:7.2f}x   {opt_rows:,}/{opt_batches:,}"
        )

        explain_blocks.append((label, sql, before, after))

    print("\n" + "=" * 60)
    print("EXPLAIN OUTPUT (unoptimized → optimized)")
    print("=" * 60 + "\n")

    for label, sql, before, after in explain_blocks:
        print(f"── {label}")
        print(f"  SQL: {sql}")
        print("  Before optimization:")
        print(_indent(before, 4))
        print("  After optimization:")
        print(_indent(after, 4))
        print()


def _indent(s: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in s.splitlines())


if __name__ == "__main__":
    main()
