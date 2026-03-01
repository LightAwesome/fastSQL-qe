import csv

import numpy as np
from catalog.table import Column, Table
from errors import QueryError


def infer_dtype(values: list[str]) -> str:
    for v in values:
        if v == "":
            raise QueryError("NULL/empty values not supported. Clean your data first.")
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
    elif dtype == "float64":
        return np.array([float(v) for v in values], dtype=np.float64)
    elif dtype == "bool":
        return np.array([v.lower() == "true" for v in values], dtype=bool)
    else:
        return np.array(values, dtype=object)


def load_csv(path: str, table_name: str = "t") -> Table:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise QueryError(f"CSV file is empty: {path}")
        if len(set(header)) != len(header):
            raise QueryError(f"Duplicate column names in CSV: {header}")
        rows = list(reader)
    if len(rows) == 0:
        columns = {h: Column(h, "object", np.array([], dtype=object)) for h in header}
        return Table(table_name, columns)
    col_data = {header[i]: [row[i] for row in rows] for i in range(len(header))}
    columns = {}
    for name, values in col_data.items():
        dtype = infer_dtype(values)
        columns[name] = Column(name, dtype, cast_column(values, dtype))
    return Table(table_name, columns)
