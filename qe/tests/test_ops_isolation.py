import numpy as np

from qe.catalog.table import Column, Table
from qe.exec.ops import FilterOp, LimitOp, ProjectOp, ScanOp
from qe.sql.ast import BinOp, ColRef, Literal, SelectItem


def make_big_table(n=10000):
    return Table(
        "t",
        {
            "id": Column("id", "int64", np.arange(n, dtype=np.int64)),
            "value": Column("value", "float64", np.linspace(0, 1, n, dtype=np.float64)),
        },
    )


def test_scan_batches_sizes():
    t = make_big_table(10)
    op = ScanOp(t, batch_size=4)
    sizes = [len(b["id"]) for b in op.batches()]
    assert sizes == [4, 4, 2]


def test_limit_op_stops_early():
    t = make_big_table(10000)
    scan = ScanOp(t, batch_size=4096)
    limited = LimitOp(scan, 10)
    out = []
    for b in limited.batches():
        out.append(b["id"])
    ids = np.concatenate(out).tolist()
    assert ids == list(range(10))


def test_filter_then_project():
    t = make_big_table(10)
    scan = ScanOp(t, batch_size=10)
    pred = BinOp(">", ColRef("value"), Literal(0.5, "float64"))
    filtered = FilterOp(scan, pred)

    proj_items = [SelectItem(expr=ColRef("id"), alias=None)]
    proj = ProjectOp(filtered, proj_items, source_table=t)

    out_batches = list(proj.batches())
    ids = out_batches[0]["id"].tolist()
    assert ids == [5, 6, 7, 8, 9]
