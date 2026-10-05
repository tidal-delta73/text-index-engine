"""Query lexer, parser and evaluator for ``search``.

Grammar (no implicit operators; precedence NOT > AND > OR):

    or    := and (OR and)*
    and   := not (AND not)*
    not   := NOT not | primary
    primary := "(" or ")" | phrase | term
    phrase := '"' ... '"'        (terms must appear consecutively)
    term  := bare word           (normalized and tokenized like document text)

A bare word that tokenizes to more than one term is matched like a phrase.
"""
from dataclasses import dataclass

from .snapshot import Snapshot
from .tokenizer import tokenize

OPERATORS = {"AND", "OR", "NOT"}


class QueryError(Exception):
    """A query lexical or syntax error (reported with exit code 2)."""


@dataclass
class Token:
    kind: str  # "AND", "OR", "NOT", "LPAREN", "RPAREN", "TERM", "PHRASE"
    terms: list[str] | None = None  # for TERM and PHRASE


def lex(query: str) -> list[Token]:
    tokens: list[Token] = []
    word: list[str] = []

    def flush_word() -> None:
        if not word:
            return
        text = "".join(word)
        word.clear()
        if text in OPERATORS:
            tokens.append(Token(text))
            return
        terms = tokenize(text)
        if not terms:
            raise QueryError(f"unknown token: {text!r}")
        tokens.append(Token("TERM", terms))

    i = 0
    while i < len(query):
        ch = query[i]
        if ch.isspace():
            flush_word()
        elif ch == "(":
            flush_word()
            tokens.append(Token("LPAREN"))
        elif ch == ")":
            flush_word()
            tokens.append(Token("RPAREN"))
        elif ch == '"':
            flush_word()
            end = query.find('"', i + 1)
            if end == -1:
                raise QueryError("unpaired quote")
            terms = tokenize(query[i + 1 : end])
            if not terms:
                raise QueryError("phrase contains no terms")
            tokens.append(Token("PHRASE", terms))
            i = end
        else:
            word.append(ch)
        i += 1
    flush_word()
    return tokens


# AST nodes: ("term", terms) | ("and", a, b) | ("or", a, b) | ("not", a)


def parse(tokens: list[Token]):
    """Parse tokens into an AST. Returns None for an empty query."""
    if not tokens:
        return None
    node, pos = _parse_or(tokens, 0)
    if pos != len(tokens):
        raise QueryError("missing operator between operands")
    return node


def _parse_or(tokens: list[Token], pos: int):
    node, pos = _parse_and(tokens, pos)
    while pos < len(tokens) and tokens[pos].kind == "OR":
        right, pos = _parse_and(tokens, pos + 1)
        node = ("or", node, right)
    return node, pos


def _parse_and(tokens: list[Token], pos: int):
    node, pos = _parse_not(tokens, pos)
    while pos < len(tokens) and tokens[pos].kind == "AND":
        right, pos = _parse_not(tokens, pos + 1)
        node = ("and", node, right)
    return node, pos


def _parse_not(tokens: list[Token], pos: int):
    if pos < len(tokens) and tokens[pos].kind == "NOT":
        node, pos = _parse_not(tokens, pos + 1)
        return ("not", node), pos
    return _parse_primary(tokens, pos)


def _parse_primary(tokens: list[Token], pos: int):
    if pos >= len(tokens):
        raise QueryError("missing operand")
    token = tokens[pos]
    if token.kind in {"TERM", "PHRASE"}:
        return ("term", token.terms), pos + 1
    if token.kind == "LPAREN":
        node, pos = _parse_or(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos].kind != "RPAREN":
            raise QueryError("unpaired parenthesis")
        return node, pos + 1
    if token.kind == "RPAREN":
        raise QueryError("unpaired parenthesis")
    raise QueryError(f"missing operand before {token.kind}")


def evaluate(node, snapshot: Snapshot) -> list[str]:
    """Evaluate an AST against a snapshot; return matching ids sorted by code point."""
    universe = set(range(len(snapshot.documents)))
    matches = _eval(node, snapshot, universe)
    return sorted(snapshot.documents[i] for i in matches)


def _eval(node, snapshot: Snapshot, universe: set[int]) -> set[int]:
    kind = node[0]
    if kind == "term":
        return _match_terms(node[1], snapshot)
    if kind == "and":
        return _eval(node[1], snapshot, universe) & _eval(node[2], snapshot, universe)
    if kind == "or":
        return _eval(node[1], snapshot, universe) | _eval(node[2], snapshot, universe)
    if kind == "not":
        return universe - _eval(node[1], snapshot, universe)
    raise AssertionError(f"unknown node: {kind}")


def _match_terms(terms: list[str], snapshot: Snapshot) -> set[int]:
    """Documents where all terms appear at consecutive positions."""
    postings = [snapshot.terms.get(term) for term in terms]
    if any(p is None for p in postings):
        return set()
    candidates = {doc_index for doc_index, _ in postings[0]}
    for posting in postings[1:]:
        candidates &= {doc_index for doc_index, _ in posting}
    if len(terms) == 1:
        return candidates
    position_sets = [dict(posting) for posting in postings]
    result: set[int] = set()
    for doc_index in candidates:
        starts = position_sets[0][doc_index]
        rest = [set(pos[doc_index]) for pos in position_sets[1:]]
        for start in starts:
            if all(start + offset + 1 in rest[offset] for offset in range(len(rest))):
                result.add(doc_index)
                break
    return result
