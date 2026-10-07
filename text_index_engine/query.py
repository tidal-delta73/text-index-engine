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
    """Iterative equivalent of recursive descent for the grammar above.

    The grammar's only recursive productions are NOT chains and
    parenthesized groups; both are handled here with explicit stacks, so
    the usable query depth is bounded by memory, not by the interpreter's
    recursion limit. The produced AST, the left associativity of AND/OR
    and the DataError cases are identical to plain recursive descent.
    """

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
        # Each '(' pushes a frame holding the enclosing or_expr's partial
        # state: (folded or_expr, folded and_expr, pending NOT count).
        # The top level is the implicit frame held in the locals below.
        frames = []
        or_node = None   # or_expr folded so far in the current frame
        and_node = None  # and_expr folded so far in the current frame
        nots = 0         # NOT keywords waiting for their operand
        expect_operand = True

        while True:
            tok = self._peek()
            kind = tok.kind
            if expect_operand:
                if kind == T_NOT:
                    self._advance()
                    nots += 1
                    continue
                if kind == T_LPAREN:
                    self._advance()
                    if self._peek().kind == T_RPAREN:
                        raise DataError(
                            "query: missing operand inside parentheses")
                    frames.append((or_node, and_node, nots))
                    or_node = and_node = None
                    nots = 0
                    continue
                if kind == T_TERM:
                    self._advance()
                    node = ("term", tok.value)
                elif kind == T_PREFIX:
                    self._advance()
                    node = ("prefix", tok.value)
                elif kind == T_PHRASE:
                    self._advance()
                    node = ("phrase", tok.value)
                elif kind == T_RPAREN:
                    raise DataError("query: unmatched ')'")
                elif kind == T_EOF:
                    raise DataError("query: missing operand")
                else:
                    raise DataError(
                        f"query: unexpected token {tok.value!r}")
            else:
                # An operand was just completed: only an infix operator, a
                # closing parenthesis or the end of the query may follow.
                if kind == T_AND:
                    self._advance()
                    expect_operand = True
                    continue
                if kind == T_OR:
                    self._advance()
                    or_node = (and_node if or_node is None
                               else ("or", or_node, and_node))
                    and_node = None
                    expect_operand = True
                    continue
                if kind == T_RPAREN:
                    if not frames:
                        raise DataError("query: unmatched ')'")
                    self._advance()
                    node = (and_node if or_node is None
                            else ("or", or_node, and_node))
                    or_node, and_node, nots = frames.pop()
                    # The group is one completed operand of the outer
                    # frame; fall through to deliver it.
                elif kind == T_EOF:
                    if frames:
                        raise DataError("query: unbalanced parentheses")
                    return (and_node if or_node is None
                            else ("or", or_node, and_node))
                else:
                    # Another operand starts where an operator was due.
                    # Recursive descent only reports adjacency at the top
                    # level; inside parentheses the same situation surfaces
                    # when the group's closing ')' is not found.
                    if frames:
                        raise DataError("query: unbalanced parentheses")
                    raise DataError(
                        "query: missing operator between operands")
            # Deliver a completed operand into the current frame: pending
            # NOTs wrap it innermost-first, then it extends the AND chain.
            for _ in range(nots):
                node = ("not", node)
            nots = 0
            and_node = node if and_node is None else ("and", and_node, node)
            expect_operand = False


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


# Apply-markers for the iterative evaluator: when a composite node is
# popped, its marker goes back on the work stack behind its children, and
# combining happens once every child result sits on the value stack.
_APPLY_NOT = ("apply:not",)
_APPLY_AND = ("apply:and",)
_APPLY_OR = ("apply:or",)


def evaluate(node, snapshot) -> set[str]:
    """Evaluate a parsed query against a snapshot, iteratively.

    Explicit work/value stacks replace recursion, so deep NOT chains and
    long AND/OR spines evaluate regardless of the interpreter recursion
    limit. Semantics are unchanged: NOT's universe is the full document
    set recorded in the snapshot.
    """
    universe = snapshot.all_docs
    values: list[set[str]] = []
    stack = [node]
    while stack:
        item = stack.pop()
        if item is _APPLY_NOT:
            values.append(set(universe) - values.pop())
            continue
        if item is _APPLY_AND:
            right = values.pop()
            values.append(values.pop() & right)
            continue
        if item is _APPLY_OR:
            right = values.pop()
            values.append(values.pop() | right)
            continue
        kind = item[0]
        if kind == "term":
            posting = snapshot.postings.get(item[1])
            values.append(set(posting) if posting is not None else set())
        elif kind == "prefix":
            values.append(_prefix_docs(snapshot, item[1]))
        elif kind == "phrase":
            values.append(_phrase_docs(snapshot, item[1]))
        elif kind == "not":
            stack.append(_APPLY_NOT)
            stack.append(item[1])
        elif kind == "and":
            stack.append(_APPLY_AND)
            stack.append(item[2])
            stack.append(item[1])
        elif kind == "or":
            stack.append(_APPLY_OR)
            stack.append(item[2])
            stack.append(item[1])
        else:
            raise DataError(f"query: internal error, unknown node {kind!r}")
    return values[0]
