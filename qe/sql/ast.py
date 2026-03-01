from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ColRef:
    name: str


@dataclass
class Literal:
    value: Any
    dtype: str


@dataclass
class BinOp:
    op: str
    left: Any
    right: Any


@dataclass
class UnaryOp:
    op: str
    operand: Any


@dataclass
class AggFunc:
    func: str
    arg: Any


@dataclass
class SelectItem:
    expr: Any
    alias: Optional[str] = None


@dataclass
class OrderByItem:
    expr: Any
    ascending: bool = True


@dataclass
class SelectQuery:
    select: List[SelectItem]
    from_table: str
    where: Optional[Any] = None
    group_by: List[Any] = field(default_factory=list)
    order_by: List[OrderByItem] = field(default_factory=list)
    limit: Optional[int] = None
    is_explain: bool = False
