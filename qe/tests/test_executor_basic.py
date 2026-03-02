import numpy as np
import pytest

from qe.catalog.table import Column, Table
from qe.exec.engine import execute
from qe.plan.analyzer import analyze
from qe.sql.parser import Parser


def make_table():
    return Table(
        "t",
        {
            "id": Column("id", "int64", np.array([1, 2, 3])),
            "category": Column(
                "category", "object", np.array(["A", "B", "A"], dtype=object)
            ),
            "value": Column("value", "float64", np.array([0.5, 0.7, 0.9])),
            "active": Column("active", "bool", np.array([True, False, True])),
        },
    )


def run(sql: str) -> Table:
    t = make_table()
    q = Parser(sql).parse()
    analyze(q, t)
    return execute(q, t, batch_size=2)


def test_select_projection():
    out = run("SELECT id, value FROM t")
    assert out.col_names() == ["id", "value"]
    assert out.row_count == 3
    assert out.get_column("id").data.tolist() == [1, 2, 3]
    assert out.get_column("value").data.tolist() == [0.5, 0.7, 0.9]


def test_where_filter():
    out = run("SELECT id FROM t WHERE value > 0.6")
    assert out.col_names() == ["id"]
    assert out.get_column("id").data.tolist() == [2, 3]


def test_expression_naming_is_stable_const():
    out = run("SELECT value * 2 FROM t")
    assert out.col_names() == ["value_mul_const"]
    assert out.get_column("value_mul_const").data.tolist() == [1.0, 1.4, 1.8]


def test_limit_streaming_no_order_by():
    out = run("SELECT id FROM t LIMIT 2")
    assert out.get_column("id").data.tolist() == [1, 2]


def test_order_by_blocking_then_limit():
    out = run("SELECT id, value FROM t ORDER BY value DESC LIMIT 2")
    assert out.get_column("id").data.tolist() == [3, 2]
    assert out.get_column("value").data.tolist() == [0.9, 0.7]
