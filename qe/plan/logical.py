from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from qe.sql.ast import OrderByItem, SelectItem, SelectQuery


class LogicalPlan:
    pass


@dataclass(frozen=True)
class TopN(LogicalPlan):
    child: LogicalPlan
    order_by: OrderByItem
    n: int


@dataclass(frozen=True)
class Scan(LogicalPlan):
    table_name: str
    needed_cols: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class Filter(LogicalPlan):
    child: LogicalPlan
    predicate: object


@dataclass(frozen=True)
class Project(LogicalPlan):
    child: LogicalPlan
    select: list[SelectItem]


@dataclass(frozen=True)
class Aggregate(LogicalPlan):
    child: LogicalPlan
    query: SelectQuery


@dataclass(frozen=True)
class Sort(LogicalPlan):
    child: LogicalPlan
    order_by: OrderByItem


@dataclass(frozen=True)
class Limit(LogicalPlan):
    child: LogicalPlan
    n: int
