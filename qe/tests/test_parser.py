import pytest

from qe.sql.ast import AggFunc, BinOp, ColRef, Literal, SelectQuery, UnaryOp
from qe.sql.parser import Parser


def test_parse_simple_select():
    q = Parser("SELECT id, value FROM t;").parse()
    assert isinstance(q, SelectQuery)
    assert q.from_table == "t"
    assert len(q.select) == 2
    assert isinstance(q.select[0].expr, ColRef)
    assert q.select[0].expr.name == "id"


def test_parse_where_precedence_and_over_or():
    q = Parser("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3").parse()
    # expect: OR(=(a,1), AND(=(b,2), =(c,3)))
    assert isinstance(q.where, BinOp)
    assert q.where.op == "OR"
    assert isinstance(q.where.right, BinOp)
    assert q.where.right.op == "AND"


def test_parse_parentheses_override_precedence():
    q = Parser("SELECT * FROM t WHERE (a = 1 OR b = 2) AND c = 3").parse()
    assert isinstance(q.where, BinOp)
    assert q.where.op == "AND"
    assert isinstance(q.where.left, BinOp)
    assert q.where.left.op == "OR"


def test_parse_unary_not():
    q = Parser("SELECT * FROM t WHERE NOT active").parse()
    assert isinstance(q.where, UnaryOp)
    assert q.where.op == "not"
    assert isinstance(q.where.operand, ColRef)
    assert q.where.operand.name == "active"


def test_parse_aggregates_group_by():
    q = Parser("SELECT category, SUM(value) FROM t GROUP BY category").parse()
    assert len(q.group_by) == 1
    assert isinstance(q.group_by[0], ColRef)
    assert q.group_by[0].name == "category"

    # second select item should be AggFunc
    assert isinstance(q.select[1].expr, AggFunc)
    assert q.select[1].expr.func == "SUM"
    assert isinstance(q.select[1].expr.arg, ColRef)
    assert q.select[1].expr.arg.name == "value"


def test_parse_order_by_limit():
    q = Parser("SELECT id FROM t ORDER BY value DESC LIMIT 10").parse()
    assert len(q.order_by) == 1
    assert q.order_by[0].ascending is False
    assert q.limit == 10
