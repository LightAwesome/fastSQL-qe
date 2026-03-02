import numpy as np
import pytest

from qe.catalog.table import Column, Table
from qe.exec.engine import execute
from qe.plan.analyzer import analyze
from qe.sql.parser import Parser


def run_sql(sql: str, table: Table):
    q = Parser(sql).parse()
    q = analyze(q, table)
    return execute(q, table)


def make_table():
    return Table(
        "t",
        {
            "id": Column("id", "int64", np.array([1, 2, 3])),
            "category": Column(
                "category", "object", np.array(["A", "B", "A"], dtype=object)
            ),
            "value": Column("value", "float64", np.array([0.5, 0.9, 1.2])),
            "active": Column("active", "bool", np.array([True, False, True])),
        },
    )


def test_whole_table_sum():
    table = make_table()
    out = run_sql("SELECT SUM(value) FROM t", table)
    assert (
        out.col_names() == ["expr_0"]
        or out.col_names() == ["sum"]
        or out.col_names() == ["SUM"]
    )
    assert out.row_count == 1
    col = out.get_column(out.col_names()[0]).data
    assert float(col[0]) == pytest.approx(2.6)


def test_whole_table_count_star():
    table = make_table()
    out = run_sql("SELECT COUNT(*) FROM t", table)
    col = out.get_column(out.col_names()[0]).data
    assert int(col[0]) == 3


def test_group_by_sum_deterministic_order():
    table = make_table()
    out = run_sql("SELECT category, SUM(value) FROM t GROUP BY category", table)
    assert out.col_names()[0] == "category"
    cats = list(out.get_column("category").data)
    assert cats == ["A", "B"]

    sums = out.get_column(out.col_names()[1]).data
    assert float(sums[0]) == 1.7
    assert float(sums[1]) == 0.9


def test_group_by_count_star():
    table = make_table()
    out = run_sql("SELECT category, COUNT(*) FROM t GROUP BY category", table)
    cats = list(out.get_column("category").data)
    counts = out.get_column(out.col_names()[1]).data
    assert cats == ["A", "B"]
    assert int(counts[0]) == 2
    assert int(counts[1]) == 1


def test_group_by_avg():
    table = make_table()
    out = run_sql("SELECT category, AVG(value) FROM t GROUP BY category", table)
    cats = list(out.get_column("category").data)
    avgs = out.get_column(out.col_names()[1]).data
    assert cats == ["A", "B"]
    assert float(avgs[0]) == 0.85  # (0.5 + 1.2) / 2
    assert float(avgs[1]) == 0.9


def test_exec_projection_only():
    table = make_table()
    out = run_sql("SELECT id, value FROM t", table)
    assert out.col_names() == ["id", "value"]
    np.testing.assert_array_equal(out.get_column("id").data, np.array([1, 2, 3]))
    np.testing.assert_array_equal(
        out.get_column("value").data, np.array([0.5, 0.9, 1.2])
    )


def test_exec_where_filter():
    table = make_table()
    out = run_sql("SELECT id FROM t WHERE value > 0.7", table)
    np.testing.assert_array_equal(out.get_column("id").data, np.array([2, 3]))


def test_exec_boolean_logic_precedence():
    table = make_table()
    out = run_sql("SELECT id FROM t WHERE active = true OR value > 1 AND id = 2", table)
    np.testing.assert_array_equal(out.get_column("id").data, np.array([1, 3]))


def test_exec_not_unary():
    table = make_table()
    out = run_sql("SELECT id FROM t WHERE NOT active", table)
    np.testing.assert_array_equal(out.get_column("id").data, np.array([2]))


def test_exec_computed_expr():
    table = make_table()
    out = run_sql("SELECT id, value * 2 FROM t WHERE active = true", table)
    assert out.col_names() == ["id", "expr_1"]
    np.testing.assert_array_equal(out.get_column("id").data, np.array([1, 3]))
    np.testing.assert_array_equal(out.get_column("expr_1").data, np.array([1.0, 2.4]))


def test_exec_limit():
    table = make_table()
    out = run_sql("SELECT id FROM t LIMIT 2", table)
    np.testing.assert_array_equal(out.get_column("id").data, np.array([1, 2]))


def test_exec_order_by_desc():
    table = make_table()
    out = run_sql("SELECT id, value FROM t ORDER BY value DESC", table)
    np.testing.assert_array_equal(out.get_column("id").data, np.array([3, 2, 1]))
    np.testing.assert_array_equal(
        out.get_column("value").data, np.array([1.2, 0.9, 0.5])
    )


def test_exec_empty_result_preserves_schema():
    table = make_table()
    out = run_sql("SELECT id, value FROM t WHERE value > 999", table)
    assert out.col_names() == ["id", "value"]
    assert out.row_count == 0
    assert out.get_column("id").data.shape[0] == 0
    assert out.get_column("value").data.shape[0] == 0
