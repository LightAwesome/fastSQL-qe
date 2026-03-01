from qe.errors import QueryError

from .ast import *
from .tokenizer import TT, Token, Tokenizer

# Binding powers (higher = tighter binding)
BP = {
    TT.OR: 10,
    TT.AND: 20,
    TT.EQ: 30,
    TT.NEQ: 30,
    TT.LT: 30,
    TT.LTE: 30,
    TT.GT: 30,
    TT.GTE: 30,
    TT.PLUS: 40,
    TT.MINUS: 40,
    TT.STAR: 50,
    TT.SLASH: 50,
    TT.PERCENT: 50,
}


class Parser:
    def __init__(self, text: str):
        self.tokens = Tokenizer(text).tokenize()
        self.pos = 0

    def error(self, msg):
        tok = self.current()
        raise QueryError(f"Parse error at '{tok.value}' (pos {tok.pos}): {msg}")

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek_type(self) -> TT:
        return self.tokens[self.pos].type

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TT.EOF:
            self.pos += 1
        return tok

    def expect(self, tt: TT) -> Token:
        if self.peek_type() != tt:
            self.error(f"Expected {tt.name}, got {self.peek_type().name}")
        return self.advance()

    def match(self, *types) -> bool:
        if self.peek_type() in types:
            self.advance()
            return True
        return False

    def parse(self) -> SelectQuery:
        is_explain = False
        if self.peek_type() == TT.EXPLAIN:
            self.advance()
            is_explain = True
        self.expect(TT.SELECT)
        select = self._parse_select_list()
        self.expect(TT.FROM)
        from_table = self.expect(TT.IDENT).value
        where = None
        if self.peek_type() == TT.WHERE:
            self.advance()
            where = self._parse_expr(0)
        group_by = []
        if self.peek_type() == TT.GROUP:
            self.advance()
            self.expect(TT.BY)
            group_by = self._parse_comma_list(self._parse_expr_0)
        order_by = []
        if self.peek_type() == TT.ORDER:
            self.advance()
            self.expect(TT.BY)
            order_by = self._parse_order_by_list()
        limit = None
        if self.peek_type() == TT.LIMIT:
            self.advance()
            limit = self.expect(TT.INT).value

        if self.peek_type() == TT.SEMICOLON:
            self.advance()
        if self.peek_type() != TT.EOF:
            self.error("Unexpected tokens after query")
        return SelectQuery(
            select, from_table, where, group_by, order_by, limit, is_explain
        )

    def _parse_select_list(self):
        items = []
        while True:
            if self.peek_type() == TT.STAR:
                self.advance()
                items.append(SelectItem(ColRef("*")))
            else:
                expr = self._parse_expr(0)
                alias = None
                if self.peek_type() == TT.AS:
                    self.advance()
                    alias = self.expect(TT.IDENT).value
                items.append(SelectItem(expr, alias))
            if self.peek_type() != TT.COMMA:
                break
            self.advance()
        return items

    def _parse_order_by_list(self):
        items = []
        while True:
            expr = self._parse_expr(0)
            asc = True
            if self.peek_type() == TT.DESC:
                self.advance()
                asc = False
            elif self.peek_type() == TT.ASC:
                self.advance()
            items.append(OrderByItem(expr, asc))
            if self.peek_type() != TT.COMMA:
                break
            self.advance()
        return items

    def _parse_comma_list(self, parse_fn):
        items = [parse_fn()]
        while self.peek_type() == TT.COMMA:
            self.advance()
            items.append(parse_fn())
        return items

    def _parse_expr_0(self):
        return self._parse_expr(0)

    # --- Pratt parser ---
    def _parse_expr(self, min_bp: int):
        left = self._parse_prefix()
        while True:
            tt = self.peek_type()
            bp = BP.get(tt, 0)
            if bp <= min_bp:
                break
            op_tok = self.advance()
            right = self._parse_expr(bp)
            op_str = (
                op_tok.value
                if isinstance(op_tok.value, str)
                else op_tok.type.name.lower()
            )
            left = BinOp(op_str, left, right)
        return left

    def _parse_prefix(self):
        tt = self.peek_type()
        if tt == TT.NOT:
            self.advance()
            return UnaryOp("not", self._parse_expr(25))
        if tt == TT.MINUS:
            self.advance()
            return UnaryOp("-", self._parse_expr(45))
        if tt == TT.LPAREN:
            self.advance()
            expr = self._parse_expr(0)
            self.expect(TT.RPAREN)
            return expr
        if tt == TT.INT:
            return Literal(self.advance().value, "int")
        if tt == TT.FLOAT:
            return Literal(self.advance().value, "float")
        if tt == TT.STRING:
            return Literal(self.advance().value, "str")
        if tt == TT.TRUE:
            self.advance()
            return Literal(True, "bool")
        if tt == TT.FALSE:
            self.advance()
            return Literal(False, "bool")
        if tt in (TT.COUNT, TT.SUM, TT.AVG, TT.MIN, TT.MAX):
            return self._parse_agg_func()
        if tt == TT.IDENT:
            return ColRef(self.advance().value)
        self.error(f"Unexpected token in expression: {self.current().value!r}")

    def _parse_agg_func(self):
        func = self.advance().value.upper()
        self.expect(TT.LPAREN)
        if func == "COUNT" and self.peek_type() == TT.STAR:
            self.advance()
            self.expect(TT.RPAREN)
            return AggFunc("COUNT", ColRef("*"))
        arg = self._parse_expr(0)
        self.expect(TT.RPAREN)
        return AggFunc(func, arg)
