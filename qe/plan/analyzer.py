from ..catalog.table import Table
from ..errors import QueryError
from ..sql.ast import *


def analyze(query: SelectQuery, table: Table):
    """Validate query against table schema. Raises QueryError on any issue."""
    if query.from_table != table.name:
        raise QueryError(f"Unknown table: '{query.from_table}'")

    col_names = set(table.col_names())
    has_agg = False
    has_non_agg = False

    for item in query.select:
        if isinstance(item.expr, ColRef) and item.expr.name == "*":
            if query.group_by:
                raise QueryError("SELECT * with GROUP BY is not supported")
            continue
        _check_expr(item.expr, col_names)
        if _contains_agg(item.expr):
            has_agg = True
        else:
            has_non_agg = True

    if query.group_by:
        group_col_names = set()
        for g in query.group_by:
            if not isinstance(g, ColRef):
                raise QueryError("GROUP BY only supports column references")
            if g.name not in col_names:
                raise QueryError(f"Unknown column in GROUP BY: '{g.name}'")
            group_col_names.add(g.name)

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
    elif has_agg and has_non_agg:
        raise QueryError(
            "Cannot mix aggregate and non-aggregate columns without GROUP BY"
        )

    if query.where:
        _check_expr(query.where, col_names)
        if _contains_agg(query.where):
            raise QueryError("Aggregates not allowed in WHERE clause")

    for ob in query.order_by:
        _check_expr(ob.expr, col_names)


def _check_expr(expr, col_names: set):
    if isinstance(expr, ColRef):
        if expr.name != "*" and expr.name not in col_names:
            raise QueryError(f"Unknown column: '{expr.name}'")
    elif isinstance(expr, BinOp):
        _check_expr(expr.left, col_names)
        _check_expr(expr.right, col_names)
    elif isinstance(expr, UnaryOp):
        _check_expr(expr.operand, col_names)
    elif isinstance(expr, AggFunc):
        if expr.arg != ColRef("*"):
            _check_expr(expr.arg, col_names)
    elif isinstance(expr, Literal):
        pass
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
        return [expr.name]
    if isinstance(expr, BinOp):
        return _collect_col_refs(expr.left) + _collect_col_refs(expr.right)
    if isinstance(expr, UnaryOp):
        return _collect_col_refs(expr.operand)
    return []
