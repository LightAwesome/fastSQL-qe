from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from qe.errors import QueryError


class TT(Enum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()

    # Identifiers and keywords
    IDENT = auto()

    # Keywords
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP = auto()
    BY = auto()
    ORDER = auto()
    LIMIT = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    ASC = auto()
    DESC = auto()
    AS = auto()
    EXPLAIN = auto()

    # Aggregate keywords
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()

    # Bool keywords
    TRUE = auto()
    FALSE = auto()

    # Operators
    EQ = auto()      # =
    NEQ = auto()     # !=
    LT = auto()      # <
    LTE = auto()     # <=
    GT = auto()      # >
    GTE = auto()     # >=
    PLUS = auto()    # +
    MINUS = auto()   # -
    STAR = auto()    # *
    SLASH = auto()   # /
    PERCENT = auto() # %

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()

    # End
    EOF = auto()


KEYWORDS = {
    "select": TT.SELECT,
    "from": TT.FROM,
    "where": TT.WHERE,
    "group": TT.GROUP,
    "by": TT.BY,
    "order": TT.ORDER,
    "limit": TT.LIMIT,
    "and": TT.AND,
    "or": TT.OR,
    "not": TT.NOT,
    "asc": TT.ASC,
    "desc": TT.DESC,
    "as": TT.AS,
    "explain": TT.EXPLAIN,
    "count": TT.COUNT,
    "sum": TT.SUM,
    "avg": TT.AVG,
    "min": TT.MIN,
    "max": TT.MAX,
    "true": TT.TRUE,
    "false": TT.FALSE,
}


@dataclass(frozen=True)
class Token:
    type: TT
    value: object  
    pos: int      


class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.n = len(text)
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self.pos < self.n:
            self._skip_whitespace()
            if self.pos >= self.n:
                break

            ch = self._peek()

            if ch.isdigit():
                self._read_number()
            elif ch == "'":
                self._read_string()
            elif ch.isalpha() or ch == "_":
                self._read_ident_or_keyword()
            else:
                self._read_symbol()

        self.tokens.append(Token(TT.EOF, None, self.pos))
        return self.tokens

    # --- internals ---

    def _error(self, msg: str) -> None:
        raise QueryError(f"Tokenizer error at position {self.pos}: {msg}")

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < self.n else ""

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def _skip_whitespace(self) -> None:
        while self.pos < self.n and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def _read_number(self) -> None:
        start = self.pos
        while self.pos < self.n and self.text[self.pos].isdigit():
            self.pos += 1

        # Float if we see .digits
        if self.pos < self.n and self.text[self.pos] == ".":
            dot_pos = self.pos
            self.pos += 1
            if self.pos >= self.n or not self.text[self.pos].isdigit():
                # "10." is not supported in this minimal tokenizer; treat as error
                self.pos = dot_pos
                self.tokens.append(Token(TT.INT, int(self.text[start:self.pos]), start))
                return

            while self.pos < self.n and self.text[self.pos].isdigit():
                self.pos += 1

            self.tokens.append(Token(TT.FLOAT, float(self.text[start:self.pos]), start))
        else:
            self.tokens.append(Token(TT.INT, int(self.text[start:self.pos]), start))

    def _read_string(self) -> None:
        # Consume opening quote
        self._advance()
        start_content = self.pos
        out_chars: list[str] = []

        while self.pos < self.n:
            ch = self._advance()
            if ch == "\\":  # escape
                if self.pos >= self.n:
                    self._error("Unterminated escape in string literal")
                out_chars.append(self._advance())
                continue
            if ch == "'":  # closing quote
                self.tokens.append(Token(TT.STRING, "".join(out_chars), start_content))
                return
            out_chars.append(ch)

        self._error("Unterminated string literal")

    def _read_ident_or_keyword(self) -> None:
        start = self.pos
        while self.pos < self.n and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1

        raw = self.text[start:self.pos]
        tt = KEYWORDS.get(raw.lower(), TT.IDENT)

        # For keywords, store canonical upper-case value; for IDENT keep original
        if tt != TT.IDENT:
            self.tokens.append(Token(tt, raw.upper(), start))
        else:
            self.tokens.append(Token(tt, raw, start))

    def _read_symbol(self) -> None:
        start = self.pos
        ch = self._advance()

        # Two-char operators
        if ch == "!":
            if self._peek() != "=":
                self._error("Expected '=' after '!'")
            self._advance()
            self.tokens.append(Token(TT.NEQ, "!=", start))
            return

        if ch == "<":
            if self._peek() == "=":
                self._advance()
                self.tokens.append(Token(TT.LTE, "<=", start))
            else:
                self.tokens.append(Token(TT.LT, "<", start))
            return

        if ch == ">":
            if self._peek() == "=":
                self._advance()
                self.tokens.append(Token(TT.GTE, ">=", start))
            else:
                self.tokens.append(Token(TT.GT, ">", start))
            return

        # One-char symbols
        if ch == "=":
            self.tokens.append(Token(TT.EQ, "=", start))
        elif ch == "+":
            self.tokens.append(Token(TT.PLUS, "+", start))
        elif ch == "-":
            self.tokens.append(Token(TT.MINUS, "-", start))
        elif ch == "*":
            self.tokens.append(Token(TT.STAR, "*", start))
        elif ch == "/":
            self.tokens.append(Token(TT.SLASH, "/", start))
        elif ch == "%":
            self.tokens.append(Token(TT.PERCENT, "%", start))
        elif ch == "(":
            self.tokens.append(Token(TT.LPAREN, "(", start))
        elif ch == ")":
            self.tokens.append(Token(TT.RPAREN, ")", start))
        elif ch == ",":
            self.tokens.append(Token(TT.COMMA, ",", start))
        elif ch == ";":
            self.tokens.append(Token(TT.SEMICOLON, ";", start))
        else:
            self._error(f"Unexpected character: {ch!r}")
