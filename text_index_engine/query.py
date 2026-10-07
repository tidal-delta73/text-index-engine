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


# Operator precedences: NOT > AND > OR. AND/OR are left associative; NOT is
# right associative (it binds the operand to its right).
_OP_PRECEDENCE = {T_NOT: 3, T_AND: 2, T_OR: 1}


def parse(query: str):
    """Parse a query string into an AST. Empty/whitespace-only returns None.

    Uses the shunting-yard algorithm on a flat token list, so usable query
    depth (chains of NOT, nested parentheses, long AND/OR chains) is bounded
    by memory rather than the interpreter recursion limit. The RPN produced
    is folded into the exact nested tuple tree a recursive-descent parser for
    the grammar would build: ``("not", x)``, ``("and", l, r)``,
    ``("or", l, r)`` and leaves ``("term"|"prefix"|"phrase", value)``.
    """
    tokens = _lex(query)
    if len(tokens) == 1:  # only EOF
        return None

    # Shunting-yard. ops holds pending (precedence, kind) operators plus
    # LPAREN markers. Each parens frame records the state of one parenthesised
    # group so the recursive parser's diagnostics survive at any depth:
    #   0 = no token consumed yet since "("
    #   2 = prefix tokens (NOT) consumed, operand still missing
    #   1 = a complete operand present
    ops: list[tuple] = []
    output: list[tuple] = []
    parens: list[int] = []
    expect_operand = True
    pos = 0
    n = len(tokens)
    while pos < n:
        tok = tokens[pos]
        kind = tok.kind
        if kind == T_EOF:
            break
        if kind == T_LPAREN:
            if not expect_operand:
                # Adjacency with a group: "(foo (bar))" stays inside the open
                # group and reads as unbalanced, while "foo (bar)" at the top
                # level is two operands without an operator.
                if parens:
                    raise DataError("query: unbalanced parentheses")
                raise DataError("query: missing operator between operands")
            parens.append(0)
            ops.append((0, T_LPAREN))
            pos += 1
            continue
        if kind == T_RPAREN:
            if not parens:
                raise DataError("query: unmatched ')'")
            frame = parens[-1]
            if frame == 0:
                # "()", "(())": nothing at all between the parens.
                raise DataError("query: missing operand inside parentheses")
            if expect_operand:
                # "(foo AND)", "(NOT)", "(": a prefix/binary was consumed but
                # its operand is missing; the recursive parser reaches the
                # ")" at an atom position and reports an unmatched paren.
                raise DataError("query: unmatched ')'")
            while ops[-1][1] != T_LPAREN:
                output.append(("op", ops.pop()[1]))
            ops.pop()  # discard the LPAREN marker
            parens.pop()
            if parens:
                parens[-1] = 1
            pos += 1
            continue
        if kind == T_NOT:
            if expect_operand:
                ops.append((_OP_PRECEDENCE[T_NOT], T_NOT))
                if parens and parens[-1] == 0:
                    parens[-1] = 2
                pos += 1
                continue
            # "foo NOT bar": NOT where a binary operator was expected. Inside
            # an open group the recursive parser's closing-token check reports
            # unbalanced parentheses; at top level it is missing adjacency.
            if parens:
                raise DataError("query: unbalanced parentheses")
            raise DataError("query: missing operator between operands")
        if kind in (T_AND, T_OR):
            # A binary keyword where an operand was required: "AND foo",
            # "NOT AND", "(OR bar)". The recursive atom reports the token.
            if expect_operand:
                raise DataError(f"query: unexpected token {tok.value!r}")
            incoming = _OP_PRECEDENCE[kind]
            # Left associative: flush equal-or-higher precedence pending
            # operators, stopping at the enclosing group marker. NOT (3) is
            # always higher than AND/OR, so it binds its operand first.
            while ops and ops[-1][1] != T_LPAREN and ops[-1][0] >= incoming:
                output.append(("op", ops.pop()[1]))
            ops.append((incoming, kind))
            expect_operand = True
            pos += 1
            continue
        # A leaf token (TERM/PREFIX/PHRASE).
        if not expect_operand:
            # "(foo bar)" stays inside its group and is reported there;
            # "foo bar" at top level is missing an explicit operator.
            if parens:
                raise DataError("query: unbalanced parentheses")
            raise DataError("query: missing operator between operands")
        if kind == T_TERM:
            output.append(("term", tok.value))
        elif kind == T_PREFIX:
            output.append(("prefix", tok.value))
        else:
            output.append(("phrase", tok.value))
        if parens:
            parens[-1] = 1
        expect_operand = False
        pos += 1

    if expect_operand:
        # "NOT", "foo AND", "(", "(((" , "(foo AND": an operand was still
        # missing when input ran out, exactly the recursive atom's EOF case.
        raise DataError("query: missing operand")
    if parens:
        # A complete expression with an unclosed group: "(foo", "((foo)".
        raise DataError("query: unbalanced parentheses")
    while ops:
        output.append(("op", ops.pop()[1]))

    # Fold the flat RPN into the nested AST, iteratively.
    stack: list = []
    for elem in output:
        if elem[0] == "op":
            op = elem[1]
            if op == T_NOT:
                stack.append(("not", stack.pop()))
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(("and" if op == T_AND else "or", left, right))
        else:
            stack.append(elem)

    if len(stack) != 1:
        # Defensive: operand/operator balance is enforced while scanning.
        raise DataError("query: missing operand")
    return stack[0]


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


def _leaf_docs(node, snapshot) -> set[str]:
    kind = node[0]
    if kind == "term":
        posting = snapshot.postings.get(node[1])
        return set(posting) if posting is not None else set()
    if kind == "prefix":
        return _prefix_docs(snapshot, node[1])
    if kind == "phrase":
        return _phrase_docs(snapshot, node[1])
    raise DataError(f"query: internal error, unknown node {kind!r}")


def evaluate(node, snapshot) -> set[str]:
    """Evaluate a parsed query against a snapshot, returning a doc-id set.

    The walk uses explicit stacks rather than Python call frames, so query
    depth is bounded by memory like parsing. Semantics are identical to the
    former recursive evaluator: leaves resolve to posting sets, AND/OR are
    set intersection/union with the tree's left-associative grouping, and NOT
    is the complement against the snapshot's full document set.
    """
    universe = snapshot.all_docs
    # Pending (node, phase) frames: phase 0 visits children first, phase 1
    # combines their already-computed sets. ``values`` holds the results.
    work: list[tuple] = [(node, 0)]
    values: list[set[str]] = []
    while work:
        cur, phase = work.pop()
        kind = cur[0]
        if kind in ("term", "prefix", "phrase"):
            values.append(_leaf_docs(cur, snapshot))
            continue
        if kind == "not":
            if phase == 0:
                work.append((cur, 1))
                work.append((cur[1], 0))
            else:
                values.append(set(universe) - values.pop())
            continue
        if kind in ("and", "or"):
            if phase == 0:
                # Right is pushed first so the left subtree evaluates first;
                # results then land on ``values`` in left-then-right order.
                work.append((cur, 1))
                work.append((cur[2], 0))
                work.append((cur[1], 0))
            else:
                right = values.pop()
                left = values.pop()
                values.append(left & right if kind == "and" else left | right)
            continue
        raise DataError(f"query: internal error, unknown node {kind!r}")
    return values[0]
