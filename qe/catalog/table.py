from typing import Dict

import numpy as np

from qe.errors import QueryError

SUPPORTED_DTYPES = {"int64", "float64", "bool", "object"}


class Column:
    def __init__(self, name: str, dtype: str, data: np.ndarray):
        if dtype not in SUPPORTED_DTYPES:
            raise QueryError(
                f"Unsupported dtype '{dtype}'. Must be one of {SUPPORTED_DTYPES}"
            )
        if not isinstance(data, np.ndarray):
            raise QueryError(f"Column data must be a numpy array, got {type(data)}")
        self.name = name
        self.dtype = dtype
        self.data = data


class Table:
    def __init__(self, name: str, columns: Dict[str, Column]):
        if not columns:
            raise QueryError("Table must have at least one column")

        lengths = [len(col.data) for col in columns.values()]
        if len(set(lengths)) > 1:
            raise QueryError(
                f"All columns must have equal length, got lengths: "
                f"{ {col.name: len(col.data) for col in columns.values()} }"
            )

        for k, col in columns.items():
            if not isinstance(col, Column):
                raise QueryError("Column is not of type Column Object")
            if k != col.name:
                raise QueryError(
                    f"Column key '{k}' does not match Column.name '{col.name}'"
                )

        self.name = name
        self.columns = columns
        self.row_count = lengths[0]

    def get_column(self, name: str) -> Column:
        if name not in self.columns:
            raise QueryError(f"Unknown column: '{name}'")
        return self.columns[name]

    def col_names(self):
        return list(self.columns.keys())
