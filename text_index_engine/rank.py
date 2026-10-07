"""BM25 relevance ranking over the boolean query candidate set.

Ranking reuses the boolean engine wholesale: the same snapshot, the same
parsed query and the same hit-set semantics (:func:`query.evaluate`). Only
document ids that survive that set semantics are scored, and only the
snapshot is read -- never the original documents.

Score contributions come from positive leaves -- leaves below an *even*
number of NOT nodes. A leaf below an odd NOT depth still participates in
candidate filtering but contributes nothing to any score:

* a term leaf contributes its own term once;
* a phrase leaf contributes each phrase word once, with no phrase bonus;
* a prefix leaf contributes every snapshot dictionary term it expands to;
* a term occurring several times across the contributing leaves accumulates
  by occurrence count.

Per term::

    idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

with k1 = 1.2, b = 0.75. N counts every snapshot document, including empty
ones; dl is the document's total position count; avgdl averages over every
document. Terms absent from the dictionary have tf 0 in every document and
so contribute exactly zero. Scores round to six decimals, half to even.
"""
import math
from collections import Counter
from decimal import Decimal

from .errors import DataError
from .query import evaluate

K1 = 1.2
B = 0.75


def _contributing_term_counts(node, snapshot) -> Counter:
    """Count scoring terms from leaves at even NOT depth.

    The walk is iterative (query depth is bounded by memory, like the parser
    and evaluator) and tracks NOT parity only: AND/OR/parentheses do not
    change a leaf's depth. Prefix leaves expand against the snapshot
    dictionary in codepoint order.
    """
    counts: Counter = Counter()
    work = [(node, 0)]
    postings = snapshot.postings
    while work:
        cur, not_parity = work.pop()
        kind = cur[0]
        if kind == "not":
            work.append((cur[1], not_parity ^ 1))
            continue
        if kind in ("and", "or"):
            # The left subtree is popped first; parity is identical for both.
            work.append((cur[2], not_parity))
            work.append((cur[1], not_parity))
            continue
        if not_parity & 1:
            # Odd NOT depth: the leaf filters candidates but never scores.
            continue
        if kind == "term":
            counts[cur[1]] += 1
        elif kind == "phrase":
            for word in cur[1]:
                counts[word] += 1
        elif kind == "prefix":
            for term in sorted(postings):
                if term.startswith(cur[1]):
                    counts[term] += 1
        else:
            raise DataError(f"query: internal error, unknown node {kind!r}")
    return counts


def _format_score(value: float) -> str:
    """Fixed six-decimal decimal string; half rounds to even."""
    if value == 0.0:
        # Collapse a possible negative zero; every summand is non-negative,
        # so this is only defensive.
        value = 0.0
    return format(value, ".6f")


def rank(node, snapshot) -> list[tuple[str, str]]:
    """Score the boolean hit set of ``node`` against ``snapshot``.

    Returns ``(document id, six-decimal score string)`` pairs sorted by score
    descending and, for equal scores, by document id in Unicode code point
    order. An empty query or an empty candidate set yields an empty list; an
    empty snapshot does too. When every document is empty (avgdl 0) every
    candidate scores zero and ids remain in code point order.
    """
    if node is None:
        return []
    doc_ids = snapshot.doc_ids
    n = len(doc_ids)
    if n == 0:
        return []

    hits = evaluate(node, snapshot)
    if not hits:
        return []
    ordered_hits = sorted(hits)

    # Document lengths from the position postings themselves: build guarantees
    # a scored document's positions form the continuous run 0..dl-1.
    lengths = dict.fromkeys(doc_ids, 0)
    for term_postings in snapshot.postings.values():
        for doc_id, positions in term_postings.items():
            lengths[doc_id] += len(positions)
    total_length = sum(lengths.values())

    if total_length == 0:
        return [(doc_id, "0.000000") for doc_id in ordered_hits]
    avgdl = total_length / n

    counts = _contributing_term_counts(node, snapshot)
    # Dictionary-missing terms are zero in every document; dictionary terms
    # are summed in code point order so the float total is reproducible.
    terms = sorted(term for term in counts if term in snapshot.postings)
    idf = {
        term: math.log(
            1.0 + (n - len(snapshot.postings[term]) + 0.5)
            / (len(snapshot.postings[term]) + 0.5))
        for term in terms
    }
    k1_plus_1 = K1 + 1.0

    results = []
    for doc_id in ordered_hits:
        dl = lengths[doc_id]
        length_norm = K1 * (1.0 - B + B * dl / avgdl)
        total = 0.0
        for term in terms:
            positions = snapshot.postings[term].get(doc_id)
            if positions is None:
                continue
            tf = len(positions)
            total += (counts[term]
                      * idf[term] * tf * k1_plus_1
                      / (tf + length_norm))
        results.append((doc_id, _format_score(total)))

    # Order by the displayed (rounded) score; results start out id-sorted and
    # the sort is stable, so equal scores keep code point id order.
    results.sort(key=lambda item: Decimal(item[1]), reverse=True)
    return results
