import numpy as np
import pytest

from qe.catalog.loader_csv import load_csv
from qe.errors import QueryError


def test_load_simple():
    table = load_csv("./fixtures/simple.csv", "simple")

    assert table.col_names() == ["id", "category", "value", "active"]
    assert table.row_count == 3
    assert table.get_column("id").dtype == "int64"
    assert table.get_column("category").dtype == "object"
    assert table.get_column("value").dtype == "float64"
    assert table.get_column("active").dtype == "bool"


def test_load_empty():
    with pytest.raises(QueryError, match=r"*row 2*"):
        table = load_csv("./fixtures/has_empty.csv", "simple")


def test_load_dup():
    with pytest.raises(QueryError):
        table = load_csv("./fixtures/dup_header.csv", "simple")
