import numpy as np

from qe.catalog.table import Column, Table
from qe.errors import QueryError


def test_create_tables():
    col1 = Column("id", "int64", np.array([1, 2, 3]))
    col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

    t = Table("t", {"id": col1, "value": col2}, 3)

    print(t.col_names())
    print(t.get_column("id").data)


def test_same_col_lengths():
    with pytest.raises(QueryError):
        col1 = Column("id", "int64", np.array([1, 2, 3, 4]))
        col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

        t = Table("t", {"id": col1, "value": col2}, 3)


def test_not_existing_col():
    with pytest.raises(QueryError):
        col1 = Column("id", "int64", np.array([1, 2, 3, 4]))
        col2 = Column("value", "float64", np.array([0.5, 0.7, 0.9]))

        t = Table("t", {"id": col1, "value": col2}, 3)
        t.get_column("name")
