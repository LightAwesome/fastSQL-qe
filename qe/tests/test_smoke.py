import pytest

from qe.errors import QueryError


def test_pytest_runs_and_queryerror_exists():
    with pytest.raises(QueryError):
        raise QueryError("smoke")


def test_queryerror_message():
    try:
        raise QueryError("hello")
    except QueryError as e:
        assert str(e) == "hello"
