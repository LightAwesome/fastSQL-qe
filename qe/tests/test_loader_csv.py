from pathlib import Path

import pytest

from qe.catalog.loader_csv import load_csv
from qe.errors import QueryError

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_simple():
    table = load_csv(str(FIXTURES / "simple.csv"), "simple")

    assert table.col_names() == ["id", "category", "value", "active"]
    assert table.row_count == 3
    assert table.get_column("id").dtype == "int64"
    assert table.get_column("category").dtype == "object"
    assert table.get_column("value").dtype == "float64"
    assert table.get_column("active").dtype == "bool"


def test_load_empty_reports_row_and_column():
    with pytest.raises(QueryError, match=r"row 2.*category|category.*row 2"):
        load_csv(str(FIXTURES / "has_empty.csv"), "simple")


def test_load_dup_header_raises():
    with pytest.raises(QueryError):
        load_csv(str(FIXTURES / "dup_header.csv"), "simple")
