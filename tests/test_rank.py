"""Tests for the ``rank`` command and its BM25 scorer.

The score expectations are never derived from the production scorer: a small
reference model in this file tokenizes the raw documents with ``unicodedata``
directly, walks the parsed AST's documented tuple shape with its own NOT
parity counter, and evaluates both the boolean candidate set and the BM25
total independently. All data is fixed; there is no randomness and no
dependence on dict/set iteration order (ids are sorted explicitly).
"""
import io
import json
import math
import os
import sys
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal

from text_index_engine import query
from text_index_engine.__main__ import main
from text_index_engine.errors import DataError
from text_index_engine.rank import K1, B, rank
from text_index_engine.snapshot import Snapshot, build_from_lines

DOCS = [
    {"id": "d1", "text": "alpha alpha beta"},
    {"id": "d2", "text": "alpha gamma delta"},
    {"id": "d3", "text": "beta beta beta"},
    {"id": "d4", "text": ""},
    {"id": "中", "text": "快狐 跳过 alpha"},
]

# A second, smaller corpus with hand-trackable lengths and empty-only cases.
TINY = [
    {"id": "x", "text": "a a b"},
    {"id": "y", "text": "a c"},
    {"id": "z", "text": ""},
]


def _lines(docs):
    return [json.dumps(d, ensure_ascii=False) + "\n" for d in docs]


def _ref_tokenize(text):
    tokens, run = [], []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        if unicodedata.category(ch)[0] in ("L", "N"):
            run.append(ch)
        elif run:
            tokens.append("".join(run))
            run = []
    if run:
        tokens.append("".join(run))
    return tokens


def _ref_model(docs):
    words = {d["id"]: _ref_tokenize(d["text"]) for d in docs}
    postings = {}
    for doc_id, toks in words.items():
        for pos, term in enumerate(toks):
            postings.setdefault(term, {}).setdefault(doc_id, []).append(pos)
    return {
        "words": words,
        "universe": set(words),
        "postings": postings,
        "dictionary": sorted(postings),
    }


def _ref_phrase_hits(postings, phrase_words):
    first = postings.get(phrase_words[0])
    if not first:
        return set()
    hits = set()
    for doc_id, starts in first.items():
        later = [set(postings.get(w, {}).get(doc_id, ()))
                 for w in phrase_words[1:]]
        for start in starts:
            if all(start + k in later[k - 1] for k in range(1, len(phrase_words[1:]) + 1)):
                hits.add(doc_id)
                break
    return hits


def _ref_candidates(node, model):
    kind = node[0]
    postings = model["postings"]
    if kind == "term":
        return set(postings.get(node[1], ()))
    if kind == "prefix":
        hits = set()
        for term in model["dictionary"]:
            if term.startswith(node[1]):
                hits.update(postings[term])
        return hits
    if kind == "phrase":
        return _ref_phrase_hits(postings, node[1])
    if kind == "not":
        return set(model["universe"]) - _ref_candidates(node[1], model)
    left = _ref_candidates(node[1], model)
    right = _ref_candidates(node[2], model)
    return left & right if kind == "and" else left | right


def _ref_scoring_terms(node, model, parity=0):
    """Independent NOT-parity walk -> {term: occurrence count}."""
    counts = {}

    def add(term):
        counts[term] = counts.get(term, 0) + 1

    def walk(n, p):
        k = n[0]
        if k == "not":
            walk(n[1], p ^ 1)
        elif k in ("and", "or"):
            walk(n[1], p)
            walk(n[2], p)
        elif p & 1:
            return  # odd NOT depth: filter only
        elif k == "term":
            add(n[1])
        elif k == "phrase":
            for w in n[1]:
                add(w)
        elif k == "prefix":
            for term in model["dictionary"]:
                if term.startswith(n[1]):
                    add(term)
        else:
            raise AssertionError(k)

    walk(node, parity)
    return counts


def _ref_rank(raw, docs):
    model = _ref_model(docs)
    node = query.parse(raw)
    if node is None:
        return []
    n = len(docs)
    if n == 0:
        return []
    candidates = _ref_candidates(node, model)
    if not candidates:
        return []
    dls = {doc_id: len(words) for doc_id, words in model["words"].items()}
    total_len = sum(dls.values())
    if total_len == 0:
        return [(doc_id, "0.000000") for doc_id in sorted(candidates)]
    avgdl = total_len / n
    counts = _ref_scoring_terms(node, model)
    rows = []
    for doc_id in sorted(candidates):
        dl = dls[doc_id]
        total = 0.0
        for term, mult in sorted(counts.items()):
            positions = model["postings"].get(term, {}).get(doc_id)
            if not positions:
                continue
            df = len(model["postings"][term])
            tf = len(positions)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            total += (mult * idf * tf * (K1 + 1.0)
                      / (tf + K1 * (1.0 - B + B * dl / avgdl)))
        rows.append((doc_id, f"{total:.6f}"))
    rows.sort(key=lambda r: (-Decimal(r[1]), r[0]))
    return rows


class RankReferenceTests(unittest.TestCase):
    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(_lines(DOCS)))

    def assert_ranked(self, raw, docs=DOCS):
        actual = rank(query.parse(raw), self.snap if docs is DOCS else
                      Snapshot.load(build_from_lines(_lines(docs))))
        expected = _ref_rank(raw, docs)
        self.assertEqual(actual, expected, raw)
        # Output contract: score strings are fixed six-decimal decimals.
        for _, score in actual:
            self.assertRegex(score, r"^-?\d+\.\d{6}$")
            self.assertNotIn("nan", score.lower())
            self.assertNotIn("inf", score.lower())
        # Scores descend; ties are in codepoint id order.
        decimals = [Decimal(s) for _, s in actual]
        self.assertEqual(decimals, sorted(decimals, reverse=True))
        for i in range(1, len(actual)):
            if decimals[i] == decimals[i - 1]:
                self.assertLess(actual[i - 1][0], actual[i][0])

    def test_plain_terms(self):
        self.assert_ranked("alpha")
        self.assert_ranked("beta")
        self.assert_ranked("delta")
        self.assert_ranked("missing")
        self.assert_ranked("ALPHA")       # casefold
        self.assert_ranked("快狐")

    def test_booleans(self):
        for q in ("alpha OR beta", "alpha AND beta", "alpha OR gamma",
                  "beta AND NOT alpha", "alpha OR NOT delta",
                  "(alpha OR beta) AND NOT gamma"):
            self.assert_ranked(q)

    def test_not_parity(self):
        # Odd NOT: no positive leaves -> every candidate scores zero.
        rows = rank(query.parse("NOT alpha"), self.snap)
        self.assertEqual([s for _, s in rows], ["0.000000"] * len(rows))
        self.assertEqual([i for i, _ in rows], sorted(i for i, _ in rows))
        # The candidate set itself matches boolean semantics.
        self.assertEqual(
            {i for i, _ in rows},
            query.evaluate(query.parse("NOT alpha"), self.snap))
        # Candidates are exactly: docs without alpha (中 contains alpha too).
        self.assertEqual({i for i, _ in rows}, {"d3", "d4"})
        # Even NOT restores scoring; odd NOT inside still suppresses only
        # its own subtree.
        self.assert_ranked("NOT NOT alpha")
        self.assert_ranked("alpha AND NOT beta")
        self.assert_ranked("NOT (alpha OR beta)")
        self.assert_ranked("NOT NOT NOT alpha")
        self.assert_ranked("alpha OR NOT (beta OR gamma)")
        self.assert_ranked("NOT (NOT alpha AND beta)")

    def test_phrase_contributes_each_word_no_bonus(self):
        self.assert_ranked('"alpha beta"')
        self.assert_ranked('"alpha alpha"')
        self.assert_ranked('"beta gamma"')          # no hits
        self.assert_ranked('"alpha beta" OR gamma')

    def test_prefix_expansion(self):
        self.assert_ranked("a*")
        self.assert_ranked("alph*")
        self.assert_ranked("d*")
        self.assert_ranked("快*")
        self.assert_ranked("zzz*")
        self.assert_ranked("a* OR NOT delta")
        self.assert_ranked("NOT a*")               # filter only, all zero

    def test_repeated_occurrences_accumulate(self):
        single = dict(rank(query.parse("alpha"), self.snap))
        doubled = dict(rank(query.parse("alpha OR alpha"), self.snap))
        for doc_id in ("d1", "d2", "中"):
            # The doubled query accumulates twice then rounds once, so the
            # displayed values may differ from 2x by one last-place unit.
            self.assertAlmostEqual(
                float(doubled[doc_id]), 2.0 * float(single[doc_id]), places=5)
        self.assert_ranked("alpha OR alpha")
        self.assert_ranked('"alpha alpha"')
        self.assert_ranked("a* OR a*")

    def test_cjk_and_normalization(self):
        self.assert_ranked("ＡＬＰＨＡ")
        self.assert_ranked("ＡＬ*")
        self.assert_ranked("跳过 OR 快狐")

    def test_tiny_handtrackable_corpus(self):
        for q in ("a", "b", "c", "a OR b", "NOT a", "a AND NOT b",
                  "NOT NOT a", '"a a"', "a*", "z*"):
            self.assert_ranked(q, TINY)

    def test_empty_corpus(self):
        snap = Snapshot.load(build_from_lines([]))
        self.assertEqual(rank(query.parse("NOT fox"), snap), [])
        self.assertEqual(rank(query.parse("fox"), snap), [])

    def test_only_empty_documents_avgdl_zero(self):
        docs = [{"id": "e1", "text": ""}, {"id": "e2", "text": " - ! "}]
        snap = Snapshot.load(build_from_lines(_lines(docs)))
        rows = rank(query.parse("NOT fox"), snap)
        self.assertEqual(rows, [("e1", "0.000000"), ("e2", "0.000000")])
        self.assertEqual(rank(query.parse("fox"), snap), [])

    def test_empty_query_node(self):
        self.assertIsNone(query.parse(""))
        self.assertEqual(rank(None, self.snap), [])

    def test_no_negative_zero_nan_or_infinity(self):
        # Candidates without any scoring term (via NOT) must print plain zero.
        for q in ("NOT alpha", "NOT a*", 'NOT "alpha beta"',
                  "NOT alpha OR NOT beta"):
            for _, score in rank(query.parse(q), self.snap):
                self.assertNotEqual(score, "-0.000000")
                self.assertTrue(score[0].isdigit())

    def test_documents_without_scoring_terms_place_last(self):
        # d4 (empty) is a candidate via NOT gamma but has zero score.
        rows = rank(query.parse("alpha OR NOT gamma"), self.snap)
        ids = [i for i, _ in rows]
        self.assertEqual(set(ids),
                         query.evaluate(query.parse("alpha OR NOT gamma"),
                                        self.snap))
        self.assertEqual(ids[-1], "d4")
        self.assertEqual(rows[-1][1], "0.000000")


class RankCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.docs_path = os.path.join(self.dir, "docs.jsonl")
        self.snap_path = os.path.join(self.dir, "snap.json")
        with open(self.docs_path, "w", encoding="utf-8") as fp:
            fp.writelines(_lines(DOCS))
        code = main(["build", self.docs_path, self.snap_path])
        self.assertEqual(code, 0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_rank_roundtrip_shape(self):
        code, out, err = self.run_cli("rank", self.snap_path, "alpha OR beta")
        self.assertEqual(code, 0, err)
        rows = json.loads(out)
        self.assertTrue(out.endswith("\n"))
        self.assertEqual(out.count("\n"), 1)
        ref = _ref_rank("alpha OR beta", DOCS)
        self.assertEqual(rows, [{"id": i, "score": s} for i, s in ref])
        for row in rows:
            self.assertEqual(set(row), {"id", "score"})
            self.assertRegex(row["score"], r"^\d+\.\d{6}$")

    def test_rank_ordering_score_desc_id_asc(self):
        code, out, _ = self.run_cli("rank", self.snap_path, "a*")
        self.assertEqual(code, 0)
        rows = json.loads(out)
        scores = [Decimal(r["score"]) for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for i in range(1, len(rows)):
            if scores[i] == scores[i - 1]:
                self.assertLess(rows[i - 1]["id"], rows[i]["id"])

    def test_empty_query_and_empty_candidates(self):
        for q in ("", "zzzzz"):
            code, out, err = self.run_cli("rank", self.snap_path, q)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "[]\n")

    def test_not_query_zero_scores_id_order(self):
        code, out, err = self.run_cli("rank", self.snap_path, "NOT alpha")
        self.assertEqual(code, 0, err)
        rows = json.loads(out)
        ids = [r["id"] for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(r["score"] == "0.000000" for r in rows))
        self.assertEqual(
            set(ids),
            set(json.loads(
                self.run_cli("search", self.snap_path, "NOT alpha")[1])))

    def test_byte_stable(self):
        _, out1, _ = self.run_cli("rank", self.snap_path, "alpha OR a*")
        _, out2, _ = self.run_cli("rank", self.snap_path, "alpha OR a*")
        self.assertEqual(out1, out2)
        self.assertEqual(out1.encode("utf-8"), out2.encode("utf-8"))

    def test_input_order_independent_bytes(self):
        snap_paths = [self.snap_path]
        for k, order in enumerate((list(reversed(DOCS)),
                                   DOCS[2:] + DOCS[:2]), start=1):
            path = os.path.join(self.dir, f"docs{k}.jsonl")
            snap = os.path.join(self.dir, f"snap{k}.json")
            with open(path, "w", encoding="utf-8") as fp:
                fp.writelines(_lines(order))
            self.assertEqual(main(["build", path, snap]), 0)
            snap_paths.append(snap)
        outs = [self.run_cli("rank", p, "alpha OR NOT beta")[1]
                for p in snap_paths]
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_wrong_arg_count_exit_2(self):
        code, out, err = self.run_cli("rank", self.snap_path)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))
        code, out, err = self.run_cli("rank", self.snap_path, "a", "b")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_invalid_query_exit_2_no_partial_output(self):
        for bad in ("alpha AND", "NOT", "(alpha", '"unterminated', "*",
                    "alpha,beta"):
            code, out, err = self.run_cli("rank", self.snap_path, bad)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", bad)
            self.assertTrue(err.splitlines()[0].startswith("error:"), bad)

    def test_invalid_snapshot_exit_2(self):
        bad = os.path.join(self.dir, "bad.json")
        with open(bad, "w", encoding="utf-8") as fp:
            fp.write("{not json")
        code, out, err = self.run_cli("rank", bad, "alpha")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))
        with open(self.snap_path, encoding="utf-8") as fp:
            obj = json.load(fp)
        obj["version"] = 7
        other = os.path.join(self.dir, "v7.json")
        with open(other, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
        code, out, err = self.run_cli("rank", other, "alpha")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_missing_snapshot_exit_1(self):
        code, out, err = self.run_cli(
            "rank", os.path.join(self.dir, "nope.json"), "alpha")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_empty_snapshot_file(self):
        path = os.path.join(self.dir, "empty.json")
        open(path, "wb").close()
        code, out, err = self.run_cli("rank", path, "NOT fox")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_help_advertises_rank(self):
        for arg in ("help", "-h", "--help"):
            code, out, _ = self.run_cli(arg)
            self.assertEqual(code, 0)
            self.assertIn("rank <snapshot>", out)

    def test_old_v1_snapshot_ranks_without_rebuild(self):
        obj = {
            "version": 1,
            "documents": ["d1", "d2"],
            "terms": [
                {"term": "app",
                 "postings": [{"id": "d1", "positions": [0]}]},
                {"term": "apple",
                 "postings": [
                     {"id": "d1", "positions": [1]},
                     {"id": "d2", "positions": [0]}]},
            ],
        }
        old = os.path.join(self.dir, "old.json")
        with open(old, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False)
        code, out, err = self.run_cli("rank", old, "app*")
        self.assertEqual(code, 0, err)
        rows = json.loads(out)
        self.assertEqual([r["id"] for r in rows], ["d1", "d2"])
        self.assertTrue(all(len(r["score"].split(".")[1]) == 6 for r in rows))
        # app and apple both hit d1 -> strictly higher than d2 (only apple).
        self.assertGreater(Decimal(rows[0]["score"]), Decimal(rows[1]["score"]))


class DeepRankTests(unittest.TestCase):
    """Rank walks must stay iterative like the boolean engine."""

    N = 5000

    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(_lines(DOCS)))

    def test_deep_not_chain_parity(self):
        saved = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            odd = rank(query.parse("NOT " * (self.N + 1) + "alpha"), self.snap)
            even = rank(query.parse("NOT " * self.N + "alpha"), self.snap)
        finally:
            sys.setrecursionlimit(saved)
        # Odd depth: zero-score complement, id ordered.
        self.assertTrue(all(s == "0.000000" for _, s in odd))
        self.assertEqual([i for i, _ in odd], sorted(i for i, _ in odd))
        # Even depth: ordinary positive scores, descending.
        decimals = [Decimal(s) for _, s in even]
        self.assertEqual(decimals, sorted(decimals, reverse=True))
        self.assertTrue(any(s != "0.000000" for _, s in even))

    def test_deep_query_is_not_data_error(self):
        try:
            rank(query.parse("(" * 1000 + "alpha" + ")" * 1000), self.snap)
        except DataError:
            self.fail("parenthesized query rejected by rank")


if __name__ == "__main__":
    unittest.main()
