from typing import Dict

import numpy as np

from qe.errors import QueryError


@dataclass
class Column:
    name: str
    dtype: str
    data: np.ndarray


@dataclass
class Table:
    name: str
    columns: Dict[str, Column]
    row_count: int

    def get_column(self, name: str):
        if name not in self.columns:
            raise QueryError(f"Unknown column: '{name}'")

    def col_names(self):
        return list(self.columns.keys())
