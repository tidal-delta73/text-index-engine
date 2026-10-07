"""BM25 relevance ranking over a boolean query's candidate set.

The candidate set is exactly what ``query.evaluate`` returns for the parsed
query; ranking only orders and scores those candidates, it never widens or
narrows the set. Scoring uses only the query tree's *positive* leaves — the
ones under an even number of NOT operators. Leaves under an odd NOT count
still filter the candidate set but contribute nothing to any score.

Each positive leaf contributes terms: a plain term contributes itself, a
phrase contributes each of its words (no phrase bonus), and a prefix expands
against the snapshot dictionary with every matching term contributing. The
same term occurring K times across positive leaves scores K times.

Per contributing term, with N the snapshot's document count, df the term's
document frequency, tf its frequency in the candidate document, dl the
document's position total and avgdl the mean position total over all N
documents (empty documents included)::

    idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

with fixed k1 = 1.2 and b = 0.75. A document's total is the sum over all
contributing term occurrences, rounded to six decimals (half to even) and
rendered as a fixed six-decimal string; never NaN, Infinity or negative
zero. Results sort by score descending, ties by document id in Unicode code
point order. N == 0 or an empty candidate set yields no rows; avgdl == 0
scores every candidate 0.000000.
"""
import math

from .errors import DataError
from .query import evaluate

K1 = 1.2
B = 0.75
_DECIMALS = 6


def _positive_terms(node, snapshot) -> list[str]:
    """Scoring term occurrences from leaves under an even NOT depth.

    The walk is iterative, so — like parsing and evaluation — query depth is
    bounded by memory, not the interpreter recursion limit.
    """
    terms: list[str] = []
    # Each frame is (node, negated); ``negated`` flips at every NOT.
    stack: list[tuple] = [(node, False)]
    while stack:
        cur, negated = stack.pop()
        kind = cur[0]
        if kind == "not":
            stack.append((cur[1], not negated))
            continue
        if kind in ("and", "or"):
            stack.append((cur[1], negated))
            stack.append((cur[2], negated))
            continue
        if negated:
            # Odd NOT depth: candidate filtering only, never scoring.
            continue
        if kind == "term":
            terms.append(cur[1])
        elif kind == "phrase":
            terms.extend(cur[1])
        elif kind == "prefix":
            prefix = cur[1]
            for term in snapshot.postings:
                if term.startswith(prefix):
                    terms.append(term)
        else:
            raise DataError(f"query: internal error, unknown node {kind!r}")
    return terms


def _format_score(score: float) -> str:
    """Fixed six-decimal string, half-to-even, never negative zero."""
    text = f"{score:.{_DECIMALS}f}"
    # A total is a sum of non-negative terms, so the only negative rendering
    # possible is a negative zero; normalize it away defensively.
    if text.startswith("-"):
        return "0.000000"
    return text


def rank(node, snapshot) -> list[dict]:
    """Rank the query's candidates, returning ``[{"id", "score"}, ...]``.

    ``node`` is a parsed query (see ``query.parse``); the candidate set and
    NOT semantics come from ``query.evaluate``. The list is ordered by score
    descending, ties broken by document id in code point order.
    """
    candidates = evaluate(node, snapshot)
    n = len(snapshot.doc_ids)
    if n == 0 or not candidates:
        return []

    # Document lengths are position totals; empty-text documents have zero
    # and still count toward N and avgdl.
    doc_len = {doc_id: 0 for doc_id in snapshot.doc_ids}
    for posting in snapshot.postings.values():
        for doc_id, positions in posting.items():
            doc_len[doc_id] += len(positions)
    avgdl = sum(doc_len.values()) / n

    terms = _positive_terms(node, snapshot)
    totals: dict[str, float] = {}
    if avgdl > 0:
        postings = snapshot.postings
        idf = {}
        for term in set(terms):
            df = len(postings.get(term, ()))
            idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for doc_id in candidates:
            norm = K1 * (1.0 - B + B * doc_len[doc_id] / avgdl)
            total = 0.0
            for term in terms:
                posting = postings.get(term)
                tf = len(posting[doc_id]) if posting is not None \
                    and doc_id in posting else 0
                if tf:
                    total += idf[term] * tf * (K1 + 1.0) / (tf + norm)
            totals[doc_id] = total
    else:
        # Every document is empty: no term can occur, all scores are zero.
        for doc_id in candidates:
            totals[doc_id] = 0.0

    rows = [
        (doc_id, _format_score(totals[doc_id]))
        for doc_id in candidates
    ]
    # Sort by the emitted (rounded) score so equal printed scores always tie
    # break by document id, code point order.
    rows.sort(key=lambda row: (-float(row[1]), row[0]))
    return [{"id": doc_id, "score": score} for doc_id, score in rows]
