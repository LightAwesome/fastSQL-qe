import numpy as np
import pytest

from qe.catalog.table import Column, Table
from qe.errors import QueryError
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
            "value": Column("value", "float64", np.array([0.5, 0.9, 1.2])),
            "active": Column("active", "bool", np.array([True, False, True])),
        },
    )


def test_analyze_ok_simple_where():
    table = make_table()
    q = Parser("SELECT id FROM t WHERE value > 0.7").parse()
    analyze(q, table)


def test_unknown_column_in_where_raises():
    table = make_table()
    q = Parser("SELECT id FROM t WHERE missing > 1").parse()
    with pytest.raises(QueryError):
        analyze(q, table)


def test_unknown_column_in_select_raises():
    table = make_table()
    q = Parser("SELECT missing FROM t").parse()
    with pytest.raises(QueryError):
        analyze(q, table)


def test_aggregate_in_where_raises():
    table = make_table()
    q = Parser("SELECT id FROM t WHERE SUM(value) > 1").parse()
    with pytest.raises(QueryError):
        analyze(q, table)


def test_mixing_agg_and_nonagg_without_group_by_raises():
    table = make_table()
    q = Parser("SELECT category, SUM(value) FROM t").parse()
    with pytest.raises(QueryError):
        analyze(q, table)


def test_group_by_missing_nonagg_column_raises():
    table = make_table()
    q = Parser("SELECT category, id, SUM(value) FROM t GROUP BY category").parse()

    with pytest.raises(QueryError):
        analyze(q, table)


def test_group_by_ok():
    table = make_table()
    q = Parser("SELECT category, SUM(value) FROM t GROUP BY category").parse()
    analyze(q, table)


def test_whole_table_agg_with_constant_is_allowed():
    table = make_table()
    q = Parser("SELECT SUM(value), 1 FROM t").parse()
    analyze(q, table)


def test_sum_star_rejected():
    with pytest.raises(QueryError):
        Parser("SELECT SUM(*) FROM t").parse()
