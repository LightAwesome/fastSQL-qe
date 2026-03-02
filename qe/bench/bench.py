from __future__ import annotations

import time

import numpy as np

from qe.catalog.table import Column, Table
from qe.exec.engine import execute
from qe.plan.analyzer import analyze
from qe.sql.parser import Parser


def make_table(n: int) -> Table:
    rng = np.random.default_rng(0)
    return Table(
        "t",
        {
            "id": Column("id", "int64", np.arange(n, dtype=np.int64)),
            "value": Column("value", "float64", rng.random(n, dtype=np.float64)),
            "active": Column("active", "bool", rng.random(n) > 0.5),
        },
    )


def run(sql: str, t: Table, optimize_plan: bool) -> float:
    q = Parser(sql).parse()
    analyze(q, t)
    t0 = time.perf_counter()
    execute(q, t, batch_size=4096, optimize_plan=optimize_plan)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main():
    n = 100000000
    t = make_table(n)

    queries = [
        ("LIMIT early stop", "SELECT id FROM t LIMIT 10"),
        ("Selective filter + projection", "SELECT id FROM t WHERE value > 0.999"),
        (
            "Compute expression",
            "SELECT value * 2 FROM t WHERE active = true LIMIT 1000",
        ),
    ]

    print(f"Rows: {n:,}\n")

    for label, sql in queries:
        ms_noopt = run(sql, t, optimize_plan=False)
        ms_opt = run(sql, t, optimize_plan=True)
        print(f"{label}")
        print(f"  SQL: {sql}")
        print(f"  no-opt: {ms_noopt:.3f} ms")
        print(f"  opt:    {ms_opt:.3f} ms")
        if ms_opt > 0:
            print(f"  speedup: {ms_noopt / ms_opt:.2f}x")
        print()


if __name__ == "__main__":
    main()
