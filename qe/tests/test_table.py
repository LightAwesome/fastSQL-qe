import numpy as np
import pytest

from qe.catalog.table import Column, Table
from qe.errors import QueryError


def test_create_table_ok():
    col1 = Column("id", "int64", np.array([1, 2, 3]))
    col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

    t = Table("t", {"id": col1, "value": col2}, 3)

    assert t.name == "t"
    assert t.row_count == 3
    assert set(t.col_names()) == {"id", "value"}
    np.testing.assert_array_equal(t.get_column("id").data, np.array([1, 2, 3]))


def test_mismatched_column_lengths_raises():
    col1 = Column("id", "int64", np.array([1, 2, 3, 4]))
    col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

    with pytest.raises(QueryError):
        Table("t", {"id": col1, "value": col2}, 4)


def test_unknown_column_raises():
    col1 = Column("id", "int64", np.array([1, 2, 3]))
    col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

    t = Table("t", {"id": col1, "value": col2}, 3)

    with pytest.raises(QueryError):
        t.get_column("name")
