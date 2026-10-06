"""Boolean query language: lexer, parser, set-based evaluator.

Grammar (explicit operators only, no implicit conjunction)::

    or_expr  := and_expr (OR and_expr)*
    and_expr := not_expr (AND not_expr)*
    not_expr := NOT not_expr | atom
    atom     := TERM | PREFIX | PHRASE | LPAREN or_expr RPAREN

Precedence is NOT > AND > OR. AND/OR/NOT are keywords only when written in
exact uppercase; lowercase forms are ordinary search terms. Query terms and
phrase terms go through the same normalization/tokenization as the indexed
text. A prefix term is a bare term immediately followed by one ``*``
(e.g. ``app*``): the text before the star is normalized/tokenized like an
ordinary term and must yield exactly one non-empty token, and the star may
only appear as the final character of a bare run. Evaluation unions the
postings of every dictionary term that starts with the normalized prefix,
compared codepoint by codepoint with no further tokenization, stemming or
locale collation. Evaluation operates on sets of document ids; NOT's
universe is the full document set recorded in the snapshot.
"""
from .analysis import tokenize
from .errors import DataError

# Token kinds
T_TERM = "TERM"
T_PREFIX = "PREFIX"
T_PHRASE = "PHRASE"
T_AND = "AND"
T_OR = "OR"
T_NOT = "NOT"
T_LPAREN = "LPAREN"
T_RPAREN = "RPAREN"
T_EOF = "EOF"

_KEYWORDS = {"AND": T_AND, "OR": T_OR, "NOT": T_NOT}


class _Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value


def _lex(query: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token(T_LPAREN, ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Token(T_RPAREN, ch))
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and query[j] != '"':
                j += 1
            if j >= n:
                raise DataError("query: unterminated quoted phrase")
            words = tokenize(query[i + 1:j])
            if not words:
                raise DataError(
                    "query: quoted phrase contains no searchable term")
            tokens.append(_Token(T_PHRASE, words))
            i = j + 1
            continue
        # Bare run up to the next delimiter.
        start = i
        while i < n and not query[i].isspace() and query[i] not in '()"':
            i += 1
        run = query[start:i]
        if "*" in run:
            # A star is legal only as the single trailing character of a
            # bare prefix term. AND*/OR*/NOT* take this path too: a starred
            # keyword is a prefix term, while bare AND/OR/NOT stay operators.
            if run.count("*") != 1 or not run.endswith("*"):
                raise DataError(
                    f"query: prefix {run!r} must be a single term followed "
                    f"by one trailing '*'")
            words = tokenize(run[:-1])
            if len(words) != 1:
                raise DataError(
                    f"query: prefix {run!r} must contain exactly one "
                    f"non-empty searchable term")
            tokens.append(_Token(T_PREFIX, words[0]))
            continue
        kind = _KEYWORDS.get(run)
        if kind is not None:
            tokens.append(_Token(kind, run))
            continue
        words = tokenize(run)
        if len(words) == 1:
            tokens.append(_Token(T_TERM, words[0]))
        elif not words:
            raise DataError(f"query: unknown lexical unit {run!r}")
        else:
            # e.g. "foo,bar": several tokens with no explicit operator.
            raise DataError(
                f"query: {run!r} splits into multiple terms without an operator")
    tokens.append(_Token(T_EOF, None))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> _Token:
        return self.tokens[self.pos]

    def _advance(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        node = self._parse_or()
        tok = self._peek()
        if tok.kind != T_EOF:
            if tok.kind == T_RPAREN:
                raise DataError("query: unmatched ')'")
            raise DataError("query: missing operator between operands")
        return node

    def _parse_or(self):
        left = self._parse_and()
        while self._peek().kind == T_OR:
            self._advance()
            right = self._parse_and()
            left = ("or", left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek().kind == T_AND:
            self._advance()
            right = self._parse_not()
            left = ("and", left, right)
        return left

    def _parse_not(self):
        tok = self._peek()
        if tok.kind == T_NOT:
            self._advance()
            return ("not", self._parse_not())
        return self._parse_atom()

    def _parse_atom(self):
        tok = self._peek()
        if tok.kind == T_TERM:
            self._advance()
            return ("term", tok.value)
        if tok.kind == T_PREFIX:
            self._advance()
            return ("prefix", tok.value)
        if tok.kind == T_PHRASE:
            self._advance()
            return ("phrase", tok.value)
        if tok.kind == T_LPAREN:
            self._advance()
            if self._peek().kind == T_RPAREN:
                raise DataError("query: missing operand inside parentheses")
            node = self._parse_or()
            closing = self._peek()
            if closing.kind != T_RPAREN:
                raise DataError("query: unbalanced parentheses")
            self._advance()
            return node
        if tok.kind == T_RPAREN:
            raise DataError("query: unmatched ')'")
        if tok.kind == T_EOF:
            raise DataError("query: missing operand")
        raise DataError(f"query: unexpected token {tok.value!r}")


def parse(query: str):
    """Parse a query string into an AST. Empty/whitespace-only returns None."""
    tokens = _lex(query)
    if len(tokens) == 1:  # only EOF
        return None
    return _Parser(tokens).parse()


def _phrase_docs(snapshot, words: list[str]) -> set[str]:
    first = snapshot.postings.get(words[0])
    if first is None:
        return set()
    candidates = set(first)
    for w in words[1:]:
        p = snapshot.postings.get(w)
        if p is None:
            return set()
        candidates &= p.keys()
        if not candidates:
            return set()
    hits: set[str] = set()
    for doc_id in candidates:
        first_positions = snapshot.postings[words[0]][doc_id]
        # Position sets for the later phrase words (sorted at build time).
        later = [set(snapshot.postings[w][doc_id]) for w in words[1:]]
        for start in first_positions:
            if all(start + k in later[k - 1] for k in range(1, len(words))):
                hits.add(doc_id)
                break
    return hits


def _prefix_docs(snapshot, prefix: str) -> set[str]:
    """Union of postings of every term starting with ``prefix``.

    The match is a plain codepoint prefix test on the normalized dictionary
    terms; an absent match yields the empty set.
    """
    hits: set[str] = set()
    for term, posting in snapshot.postings.items():
        if term.startswith(prefix):
            hits.update(posting)
    return hits


def evaluate(node, snapshot) -> set[str]:
    kind = node[0]
    universe = snapshot.all_docs
    if kind == "term":
        posting = snapshot.postings.get(node[1])
        return set(posting) if posting is not None else set()
    if kind == "prefix":
        return _prefix_docs(snapshot, node[1])
    if kind == "phrase":
        return _phrase_docs(snapshot, node[1])
    if kind == "not":
        return set(universe) - evaluate(node[1], snapshot)
    if kind == "and":
        return evaluate(node[1], snapshot) & evaluate(node[2], snapshot)
    if kind == "or":
        return evaluate(node[1], snapshot) | evaluate(node[2], snapshot)
    raise DataError(f"query: internal error, unknown node {kind!r}")
