from __future__ import annotations

from ..catalog.table import Table
from ..errors import QueryError
from ..sql.ast import *


def analyze(query: SelectQuery, table: Table) -> SelectQuery:
    """Validate query against table schema. Raises QueryError on any issue."""
    if query.from_table != table.name:
        raise QueryError(f"Unknown table: '{query.from_table}'")

    col_names = set(table.col_names())

    has_agg = False
    has_nonagg_colref = False

    # SELECT clause checks
    for item in query.select:
        if isinstance(item.expr, ColRef) and item.expr.name == "*":
            if query.group_by:
                raise QueryError("SELECT * with GROUP BY is not supported")
            continue

        _check_expr(item.expr, col_names)

        if _contains_agg(item.expr):
            has_agg = True
        else:
            refs = _collect_col_refs(item.expr)
            if refs:
                has_nonagg_colref = True

    # WHERE clause checks
    if query.where:
        _check_expr(query.where, col_names)
        if _contains_agg(query.where):
            raise QueryError("Aggregates not allowed in WHERE clause")

    # ORDER BY checks (minimal: just validate column existence inside expr)
    for ob in query.order_by:
        _check_expr(ob.expr, col_names)

    # GROUP BY checks + aggregate mixing
    if query.group_by:
        group_col_names = set()
        for g in query.group_by:
            if not isinstance(g, ColRef):
                raise QueryError("GROUP BY only supports column references")
            if g.name not in col_names:
                raise QueryError(f"Unknown column in GROUP BY: '{g.name}'")
            if g.name == "*":
                raise QueryError("GROUP BY '*' is not supported")
            group_col_names.add(g.name)

        # If there is any aggregate, enforce group rule on non-agg expressions that reference columns
        for item in query.select:
            if isinstance(item.expr, ColRef) and item.expr.name == "*":
                continue
            if not _contains_agg(item.expr):
                refs = _collect_col_refs(item.expr)
                for ref in refs:
                    if ref not in group_col_names:
                        raise QueryError(
                            f"Column '{ref}' must appear in GROUP BY or be used in an aggregate"
                        )

    else:
        # No GROUP BY: cannot mix aggregate + non-agg column refs
        if has_agg and has_nonagg_colref:
            raise QueryError(
                "Cannot mix aggregate and non-aggregate column references without GROUP BY"
            )

    return query


def _check_expr(expr, col_names: set[str]) -> None:
    if isinstance(expr, ColRef):
        if expr.name != "*" and expr.name not in col_names:
            raise QueryError(f"Unknown column: '{expr.name}'")

    elif isinstance(expr, BinOp):
        _check_expr(expr.left, col_names)
        _check_expr(expr.right, col_names)

    elif isinstance(expr, UnaryOp):
        _check_expr(expr.operand, col_names)

    elif isinstance(expr, AggFunc):
        # COUNT(*)
        if isinstance(expr.arg, ColRef) and expr.arg.name == "*":
            if expr.func != "COUNT":
                raise QueryError(
                    f"{expr.func}(*) is not supported (only COUNT(*) allowed)"
                )
            return
        _check_expr(expr.arg, col_names)

    elif isinstance(expr, Literal):
        return

    else:
        raise QueryError(f"Unknown expression type: {type(expr)}")


def _contains_agg(expr) -> bool:
    if isinstance(expr, AggFunc):
        return True
    if isinstance(expr, BinOp):
        return _contains_agg(expr.left) or _contains_agg(expr.right)
    if isinstance(expr, UnaryOp):
        return _contains_agg(expr.operand)
    return False


def _collect_col_refs(expr) -> list[str]:
    if isinstance(expr, ColRef):
        return [] if expr.name == "*" else [expr.name]
    if isinstance(expr, BinOp):
        return _collect_col_refs(expr.left) + _collect_col_refs(expr.right)
    if isinstance(expr, UnaryOp):
        return _collect_col_refs(expr.operand)
    if isinstance(expr, AggFunc):
        if isinstance(expr.arg, ColRef) and expr.arg.name == "*":
            return []
        return _collect_col_refs(expr.arg)
    return []
