import csv

import numpy as np

from qe.catalog.table import Column, Table
from qe.errors import QueryError


def infer_dtype(values: list[str]) -> str:
    # assumes empties already checked at load time
    try:
        [int(v) for v in values]
        return "int64"
    except ValueError:
        pass
    try:
        [float(v) for v in values]
        return "float64"
    except ValueError:
        pass
    if all(v.lower() in ("true", "false") for v in values):
        return "bool"
    return "object"


def cast_column(values: list[str], dtype: str) -> np.ndarray:
    if dtype == "int64":
        return np.array([int(v) for v in values], dtype=np.int64)
    if dtype == "float64":
        return np.array([float(v) for v in values], dtype=np.float64)
    if dtype == "bool":
        return np.array([v.lower() == "true" for v in values], dtype=bool)
    return np.array(values, dtype=object)


def load_csv(path: str, table_name: str = "t") -> Table:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise QueryError(f"CSV file is empty: {path}")
        if len(set(header)) != len(header):
            raise QueryError(f"Duplicate column names in CSV: {header}")

        rows = []
        for data_row_idx, row in enumerate(reader, start=1):  # 1-based data rows
            if len(row) != len(header):
                raise QueryError(
                    f"Row {data_row_idx} has {len(row)} fields but header has {len(header)}"
                )
            for j, cell in enumerate(row):
                cell = cell.strip()
                if cell == "":
                    raise QueryError(
                        f"Empty cell at row {data_row_idx}, column '{header[j]}'"
                    )
            rows.append(row)

    if len(rows) == 0:
        columns = {h: Column(h, "object", np.array([], dtype=object)) for h in header}
        return Table(table_name, columns)

    # transpose rows -> columns, preserving header order
    columns = {}
    for j, col_name in enumerate(header):
        values = [row[j] for row in rows]
        dtype = infer_dtype(values)
        columns[col_name] = Column(col_name, dtype, cast_column(values, dtype))

    return Table(table_name, columns)
