import pytest

from qe.errors import QueryError
from qe.sql.tokenizer import TT, Tokenizer


def types(tokens):
    return [t.type for t in tokens]


def values(tokens):
    return [t.value for t in tokens]


def test_tokenize_basic_select():
    toks = Tokenizer("SELECT id, value FROM t;").tokenize()
    assert types(toks) == [
        TT.SELECT,
        TT.IDENT,
        TT.COMMA,
        TT.IDENT,
        TT.FROM,
        TT.IDENT,
        TT.SEMICOLON,
        TT.EOF,
    ]
    assert values(toks)[:6] == ["SELECT", "id", ",", "value", "FROM", "t"]


def test_tokenize_keywords_case_insensitive():
    toks = Tokenizer("select ID FrOm t").tokenize()
    assert types(toks) == [TT.SELECT, TT.IDENT, TT.FROM, TT.IDENT, TT.EOF]
    assert values(toks)[:4] == ["SELECT", "ID", "FROM", "t"]


def test_tokenize_numbers_int_and_float():
    toks = Tokenizer("SELECT * FROM t WHERE a = 10 AND b = 10.5").tokenize()

    assert any(t.type == TT.INT and t.value == 10 for t in toks)
    assert any(t.type == TT.FLOAT and t.value == 10.5 for t in toks)


def test_tokenize_string_literal_with_space():
    toks = Tokenizer("SELECT * FROM t WHERE name = 'hello world'").tokenize()
    assert any(t.type == TT.STRING and t.value == "hello world" for t in toks)


def test_tokenize_bool_literals():
    toks = Tokenizer("SELECT * FROM t WHERE active = true OR active = FALSE").tokenize()
    assert any(t.type == TT.TRUE for t in toks)
    assert any(t.type == TT.FALSE for t in toks)


def test_tokenize_multi_char_operators():
    toks = Tokenizer("SELECT * FROM t WHERE a>=10 AND b!=3 AND c<=5").tokenize()
    assert TT.GTE in types(toks)
    assert TT.NEQ in types(toks)
    assert TT.LTE in types(toks)


def test_tokenize_unterminated_string_raises():
    with pytest.raises(QueryError):
        Tokenizer("SELECT * FROM t WHERE name = 'oops").tokenize()


def test_tokenize_unexpected_character_raises():
    with pytest.raises(QueryError):
        Tokenizer("SELECT $ FROM t").tokenize()
