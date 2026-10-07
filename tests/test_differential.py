"""Deterministic differential tests for the query parser and set evaluator.

The production query machinery is never used to derive expected answers. A
small test-side reference model (it imports :mod:`text_index_engine.analysis`
*only* for the shared NFKC/casefold/tokenization specification -- never
``query.parse`` / ``query.evaluate``) recomputes every matching set directly
from the original documents' normalized token sequences:

* a term matches the documents whose normalized sequence contains it;
* a phrase requires the words at *consecutive* positions;
* a prefix is a plain codepoint-by-codepoint prefix over the normalized
  dictionary terms (an unmatched prefix is the empty set);
* ``NOT``'s universe is the *entire* snapshot document set, including the
  empty-text documents;
* the boolean precedence is ``NOT`` > ``AND`` > ``OR``, left associative.

The corpus, the query set, every input permutation and the assertion order
are all fixed literals / deterministic enumerations: there is no random
seed, no locale dependence, no hash-traversal reliance and no third-party
library. Each legal query is checked at the module entry point
(``parse``/``evaluate`` against an in-memory snapshot) and at the CLI entry
point, whose unique representation is the compact, code-point-sorted JSON
array plus one trailing newline.
"""
import io
import itertools
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from text_index_engine import query
from text_index_engine.analysis import tokenize
from text_index_engine.__main__ import main
from text_index_engine.errors import DataError
from text_index_engine.snapshot import Snapshot, build_from_lines

# ---------------------------------------------------------------------------
# Fixed, bounded corpus.
#
# Eight small documents packing in every required phenomenon:
#   * empty / punctuation-only text ............ d01, d02
#   * repeated words inside one document ....... d03 (fox x4), d04 (quick x2)
#   * Unicode compatibility characters ......... d04 (ligature, fullwidth,
#                                                      circled digits)
#   * case variants ............................ d04, d05, d07
#   * accent divergence in a prefix family ..... cafe (d05) vs café (d04,d07)
#   * shared prefix families ................... cat/catch/category,
#                                                 cafe/café, fox/foxglove,
#                                                 j/jump, CJK 狐/狐狸
#   * repeated consecutive phrases across docs . "quick fox" in d04,d05,d08
#
# Normalized token sequences (NFKC + casefold, then L/N runs):
#   d03: fox fox fox fox
#   d04: quick quick fox café file 12 cat catch
#   d05: quick fox cafe
#   d06: category foxglove jump
#   d07: 狐 狐狸 café cat
#   d08: quick fox
# ---------------------------------------------------------------------------
CORPUS_TAG = "fixed-corpus-v1"

DOCS = [
    {"id": "d01", "text": ""},
    {"id": "d02", "text": "  ...  !!!  "},
    {"id": "d03", "text": "fox fox fox fox"},
    {"id": "d04", "text": "QUICK quick fox café ﬁle ①２ CAT catch"},
    {"id": "d05", "text": "Quick Fox Cafe"},
    {"id": "d06", "text": "category foxglove jump"},
    {"id": "d07", "text": "狐 狐狸 café CAT"},
    {"id": "d08", "text": "quick fox"},
]

# Four deterministic input orders (fixed index lists -- never set iteration).
INPUT_ORDERS = [
    ("canonical", DOCS),
    ("reversed", list(reversed(DOCS))),
    ("rotated", DOCS[3:] + DOCS[:3]),
    ("interleaved", [DOCS[i] for i in (3, 0, 6, 1, 5, 2, 7, 4)]),
]


def _jsonl(docs):
    return [json.dumps(d, ensure_ascii=False) + "\n" for d in docs]


# ---------------------------------------------------------------------------
# Independent reference semantics, computed straight from the raw documents.
# It shares only the analyzer specification; it never imports query.py.
# ---------------------------------------------------------------------------
class ReferenceModel:
    """Set semantics over normalized per-document token lists."""

    def __init__(self, docs):
        self.tokens = {d["id"]: tokenize(d["text"]) for d in docs}
        self.universe = frozenset(self.tokens)  # NOT's universe: every doc
        self.dictionary = frozenset(
            tok for seq in self.tokens.values() for tok in seq)

    def term_docs(self, term):
        return frozenset(
            doc for doc, seq in self.tokens.items() if term in seq)

    def phrase_docs(self, words):
        words = tuple(words)
        n = len(words)
        hits = set()
        for doc, seq in self.tokens.items():
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n]) == words:
                    hits.add(doc)
                    break
        return frozenset(hits)

    def prefix_docs(self, prefix):
        # Codepoint-by-codepoint prefix on normalized dictionary terms only.
        hits = set()
        for term in self.dictionary:
            if term.startswith(prefix):
                hits.update(self.term_docs(term))
        return frozenset(hits)

    def leaf(self, node):
        kind = node[0]
        if kind == "term":
            return self.term_docs(node[1])
        if kind == "phrase":
            return self.phrase_docs(node[1])
        if kind == "prefix":
            return self.prefix_docs(node[1])
        raise AssertionError(f"reference: unknown leaf {node!r}")

    def eval_tree(self, node):
        kind = node[0]
        if kind in ("term", "phrase", "prefix"):
            return self.leaf(node)
        if kind == "not":
            return frozenset(self.universe - self.eval_tree(node[1]))
        if kind == "and":
            return self.eval_tree(node[1]) & self.eval_tree(node[2])
        if kind == "or":
            return self.eval_tree(node[1]) | self.eval_tree(node[2])
        raise AssertionError(f"reference: unknown node {node!r}")


# ---------------------------------------------------------------------------
# Deterministic legal-query enumeration.
#
# A pair is (tree, raw): tree is the exact semantic expression the reference
# evaluates; raw is the literal string given to the production parser. Every
# collection is a fixed tuple/list; enumeration uses product / fixed slicing,
# and the final list is sorted and de-duplicated deterministically.
# ---------------------------------------------------------------------------

# (normalized term, raw spelling). Raw spellings exercise normalization:
# ligature, fullwidth, circled digits and case variants.
TERM_CASES = [
    ("fox", "fox"),
    ("quick", "QUICK"),
    ("café", "CAFÉ"),     # accent stays after casefold; matches d04,d07
    ("cafe", "cafe"),     # no accent; a different term, matches d05 only
    ("file", "ﬁle"),     # ligature NFKC -> file
    ("cat", "cat"),
    ("catch", "catch"),
    ("category", "category"),
    ("foxglove", "foxglove"),
    ("jump", "jump"),
    ("狐", "狐"),
    ("狐狸", "狐狸"),
    ("12", "①２"),        # circled 1 + fullwidth 2 -> "12"
    ("missing", "missing"),
]

# (normalized word tuple, raw quoted spelling). Several raw spellings differ
# in case/separators yet denote the same phrase; two denote the empty set.
PHRASE_CASES = [
    (("quick", "fox"), '"quick fox"'),
    (("quick", "fox"), '"QUICK    fox"'),
    (("fox", "fox"), '"fox fox"'),
    (("cat", "catch"), '"CAT catch"'),
    (("café", "cat"), '"café CAT"'),
    (("狐", "狐狸"), '"狐 狐狸"'),
    (("quick",), '"QUICK"'),
    (("file", "12"), '"ﬁle ①２"'),
    (("quick", "fox", "jump"), '"quick fox jump"'),   # never consecutive
    (("fox", "cat"), '"fox cat"'),                    # no such adjacency
]

# (normalized prefix, raw spelling): matched, unmatched, normalized, CJK.
PREFIX_CASES = [
    ("fox", "fox*"),       # fox + foxglove
    ("fox", "FOX*"),       # casefolded
    ("ca", "ca*"),         # cat/catch/category/cafe/café
    ("cat", "cat*"),       # cat/catch/category
    ("caf", "CAF*"),       # cafe + café
    ("cafe", "cafe*"),     # term "cafe" only; café does not start with cafe
    ("f", "f*"),           # file/fox/foxglove
    ("j", "j*"),           # jump
    ("狐", "狐*"),         # 狐 and 狐狸
    ("狸", "狸*"),         # second codepoint only -> nothing
    ("qu", "ＱＵ*"),       # fullwidth -> qu
    ("zzz", "zzz*"),       # unmatched -> empty
]


def _all_leaves():
    specs = [(("term", norm), raw) for norm, raw in TERM_CASES]
    specs += [(("phrase", words), raw) for words, raw in PHRASE_CASES]
    specs += [(("prefix", norm), raw) for norm, raw in PREFIX_CASES]
    return tuple(specs)


# The cartesian product is taken over this smaller but coverage-dense
# operand set: ordinary terms, a single-doc term, an unmatched term, the
# accent-divergent pair, Unicode/CJK leaves, and matched/empty prefixes and
# phrases. All leaves still appear standalone and under NOT.
_BINARY_OPERANDS = [
    ("term", "fox"),
    ("term", "cafe"),
    ("term", "cat"),
    ("term", "jump"),
    ("term", "missing"),
    ("term", "12"),
    ("term", "狐狸"),
    ("prefix", "cat"),
    ("prefix", "caf"),
    ("prefix", "zzz"),
    ("phrase", ("quick", "fox")),
    ("phrase", ("fox", "cat")),
]


def _struct(tree):
    """Stable hashable structural key for a tree."""
    kind = tree[0]
    if kind in ("term", "prefix"):
        return (kind, tree[1])
    if kind == "phrase":
        return ("phrase",) + tuple(tree[1])
    if kind == "not":
        return ("not", _struct(tree[1]))
    return (kind, _struct(tree[1]), _struct(tree[2]))


def _build_queries():
    leaves = _all_leaves()
    raw_by_tree = {}
    for tree, raw in leaves:
        raw_by_tree.setdefault(tree, raw)
    operands = [(tree, raw_by_tree[tree]) for tree in _BINARY_OPERANDS]

    pairs = list(leaves)                                   # standalone leaves
    pairs += [(("not", t), "NOT " + r) for t, r in leaves]
    pairs += [(("not", ("not", t)), "NOT NOT " + r)        # double negation
              for t, r in leaves[::5]]

    for (t1, r1), (t2, r2) in itertools.product(operands, repeat=2):
        pairs.append((("and", t1, t2), r1 + " AND " + r2))
        pairs.append((("or", t1, t2), r1 + " OR " + r2))
        pairs.append((("and", t1, t2), "(" + r1 + " AND " + r2 + ")"))
        pairs.append((("or", t1, t2), "(" + r1 + " OR " + r2 + ")"))
        pairs.append((("and", ("not", t1), t2),
                      "NOT " + r1 + " AND " + r2))
        pairs.append((("and", t1, ("not", t2)),
                      r1 + " AND NOT " + r2))
        pairs.append((("or", ("not", t1), t2),
                      "NOT " + r1 + " OR " + r2))

    # Flat three-operand precedence probes (NOT > AND > OR).
    probes = [
        ("fox", "cat", "jump"),
        ("quick", "fox", "café"),
        ("fox", "foxglove", "catch"),
        ("狐", "狐狸", "cat"),
        ("missing", "fox", "file"),
    ]

    def t(w):
        return ("term", w)

    for a, b, c in probes:
        ra = raw_by_tree[t(a)]
        rb = raw_by_tree[t(b)]
        rc = raw_by_tree[t(c)]
        pairs.append((("or", t(a), ("and", t(b), t(c))),
                      ra + " OR " + rb + " AND " + rc))
        pairs.append((("or", ("and", t(a), t(b)), t(c)),
                      ra + " AND " + rb + " OR " + rc))
        pairs.append((("and", ("not", t(a)), t(b)),
                      "NOT " + ra + " AND " + rb))
        pairs.append((("or", ("not", t(a)), t(b)),
                      "NOT " + ra + " OR " + rb))
        pairs.append((("or", t(a), ("and", ("not", t(b)), t(c))),
                      ra + " OR NOT " + rb + " AND " + rc))

    # Parenthesis nesting: three spellings for one tree, plus the flat
    # spelling which precedence gives a *different* tree.
    grouped = ("and", ("or", t("fox"), t("cat")),
               ("not", ("prefix", "zzz")))
    flat = ("or", t("fox"), ("and", t("cat"),
                             ("not", ("prefix", "zzz"))))
    pairs += [
        (grouped, "(fox OR cat) AND NOT zzz*"),
        (grouped, "((fox OR cat)) AND (NOT zzz*)"),
        (grouped, "( (fox OR cat) AND NOT zzz* )"),
        (flat, "fox OR cat AND NOT zzz*"),
    ]

    # Whitespace spellings that must not change meaning.
    pairs.append((t("fox"), "  fox  "))
    pairs.append((("and", t("fox"), t("cat")), "fox   AND\tcat"))
    pairs.append((("or", t("fox"), ("not", t("cat"))), "(fox) OR (NOT (cat))"))
    pairs.append((("phrase", ("quick", "fox")), '"quick\tfox"'))

    ordered, seen = [], set()
    for tree, raw in sorted(pairs, key=lambda p: (p[1], _struct(p[0]))):
        sig = (raw, _struct(tree))
        if sig in seen:
            continue
        seen.add(sig)
        ordered.append((tree, raw))
    return ordered


LEGAL_QUERIES = _build_queries()


# ---------------------------------------------------------------------------
# Deterministic invalid-query enumeration.
# ---------------------------------------------------------------------------
def _invalid_queries():
    out = []

    def add(kind, raw):
        out.append((kind, raw))

    terms = ["fox", "cat", "qu*", '"quick fox"']

    # Missing operands around binary operators / NOT / at the edges.
    for op in ("AND", "OR"):
        add("missing-operand", op)
        add("missing-operand", op + " fox")
        add("missing-operand", "fox " + op)
        add("missing-operand", "( " + op + " )")
        add("missing-operand", "fox " + op + " " + op + " cat")
    add("missing-operand", "NOT")
    add("missing-operand", "NOT NOT")
    add("missing-operand", "( fox AND )")
    add("missing-operand", "( AND fox )")
    add("missing-operand", "fox AND ( OR cat )")

    # Adjacent operands with no explicit operator.
    for a, b in itertools.product(terms, repeat=2):
        add("adjacent-operands", a + " " + b)
    for raw in ("fox NOT cat", "NOT fox cat", "(fox) cat", "fox (cat)",
                "(fox OR cat) dog", "fox AND cat dog", "NOT NOT fox cat"):
        add("adjacent-operands", raw)

    # Empty parentheses.
    for raw in ("()", "( )", "(( ))", "fox AND ()", "() OR cat",
                "NOT ()", "(())", "( () )"):
        add("empty-parentheses", raw)

    # Unbalanced / unmatched parentheses.
    for raw in ("(fox", "((fox", "fox)", "(fox AND cat",
                "fox AND cat)", "((fox AND cat)",
                "(fox OR cat))", "(fox OR (cat AND dog)",
                "(fox))", ") fox", "fox AND (cat"):
        add("unbalanced-parentheses", raw)

    # Unterminated quoted phrases.
    for raw in ('"', '"quick', '"quick fox', 'fox AND "quick',
                '"fox" AND "unterminated', 'NOT "quick fox'):
        add("unterminated-quote", raw)

    # Illegal star placement.
    for raw in ("*", "**", "***", "fox**", "*fox", "fox*bar", "fo*o*",
                "a*b*", "foo *", "* foo", "!!!*", "-*", "*,",
                "fox,bar*", "quick-fox*", "(fox)*", "fox* *",
                "qu**", "FOX*bar", "fox***"):
        add("illegal-star", raw)

    # Phrases with no searchable term and bare runs with no lexical unit.
    for raw in ('"!!!"', '"..."', '"***"', '" - "', "...", "---",
                "fox,dog", '"quick*" AND "!!!"'):
        add("no-lexical-unit", raw)

    return out


INVALID_QUERIES = _invalid_queries()


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
class DifferentialFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reference = ReferenceModel(DOCS)
        cls.snapshots = {}
        cls.snapshot_bytes = {}
        for name, docs in INPUT_ORDERS:
            raw = build_from_lines(_jsonl(docs))
            cls.snapshot_bytes[name] = raw
            cls.snapshots[name] = Snapshot.load(raw)


class SnapshotOrderInvarianceTests(DifferentialFixture):
    def test_all_input_orders_produce_identical_bytes(self):
        canonical = self.snapshot_bytes["canonical"]
        for name, _docs in INPUT_ORDERS:
            self.assertEqual(
                self.snapshot_bytes[name], canonical,
                msg=(f"corpus={CORPUS_TAG} order={name!r}: snapshot bytes "
                     f"differ from canonical"),
            )

    def test_rebuild_is_a_pure_function_of_the_document_set(self):
        self.assertEqual(build_from_lines(_jsonl(DOCS)),
                         self.snapshot_bytes["canonical"])


class LegalQueryDifferentialTests(DifferentialFixture):
    """Module-level parse/evaluate vs the independent reference model."""

    def _assert_order(self, snap_name):
        snap = self.snapshots[snap_name]
        for tree, raw in LEGAL_QUERIES:
            expected = frozenset(self.reference.eval_tree(tree))
            try:
                actual = frozenset(query.evaluate(query.parse(raw), snap))
            except Exception as exc:  # report with full reconciliation context
                self.fail(
                    f"corpus={CORPUS_TAG} order={snap_name!r} query={raw!r}\n"
                    f"expected(sorted)={sorted(expected)}\n"
                    f"actual=<raised {type(exc).__name__}: {exc}>"
                )
            if actual != expected:
                self.fail(
                    f"corpus={CORPUS_TAG} order={snap_name!r} query={raw!r}\n"
                    f"expected(sorted)={sorted(expected)}\n"
                    f"actual(sorted)={sorted(actual)}"
                )

    def test_canonical_order_matches_reference(self):
        self._assert_order("canonical")

    def test_every_input_order_matches_reference(self):
        for name, _docs in INPUT_ORDERS:
            self._assert_order(name)

    def test_enumeration_is_deterministic_and_covers_families(self):
        self.assertGreater(len(LEGAL_QUERIES), 800)
        raws = [raw for _t, raw in LEGAL_QUERIES]
        self.assertEqual(raws, sorted(raws))  # fixed assertion order
        joined = "\x00".join(raws)
        for needle in ("*", '"', "NOT", "AND", "OR", "(", "ﬁ", "狐", "①"):
            self.assertIn(needle, joined)


class CliDifferentialTests(DifferentialFixture):
    """CLI: compact codepoint-sorted JSON + newline, identical across orders."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.paths = {}
        for name, docs in INPUT_ORDERS:
            jsonl_path = os.path.join(cls.dir, f"docs-{name}.jsonl")
            snap_path = os.path.join(cls.dir, f"snap-{name}.json")
            with open(jsonl_path, "w", encoding="utf-8") as fp:
                fp.writelines(_jsonl(docs))
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["build", jsonl_path, snap_path])
            assert code == 0, err.getvalue()
            cls.paths[name] = snap_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _search(snap_path, q):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["search", snap_path, q])
        return code, out.getvalue(), err.getvalue()

    def test_every_query_against_every_order_matches_reference_bytes(self):
        # One loop enforces both requirements at once: for every query the
        # CLI output on every order equals the reference JSON exactly, which
        # also makes the outputs byte-identical across the input orders.
        baseline = {}
        for name, _docs in INPUT_ORDERS:
            path = self.paths[name]
            for tree, raw in LEGAL_QUERIES:
                expected = (
                    json.dumps(sorted(self.reference.eval_tree(tree)),
                               ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                code, out, err = self._search(path, raw)
                if code != 0:
                    self.fail(
                        f"corpus={CORPUS_TAG} order={name!r} query={raw!r}\n"
                        f"expected={expected!r}\n"
                        f"actual=<exit {code}, stderr={err!r}>"
                    )
                if out != expected:
                    self.fail(
                        f"corpus={CORPUS_TAG} order={name!r} query={raw!r}\n"
                        f"expected={expected!r}\nactual={out!r}"
                    )
                if name == "canonical":
                    baseline[raw] = out
                else:
                    self.assertEqual(
                        out, baseline[raw],
                        msg=(f"corpus={CORPUS_TAG} order={name!r} "
                             f"query={raw!r}\nexpected={baseline[raw]!r}\n"
                             f"actual={out!r}"),
                    )

    def test_snapshot_files_are_byte_identical_across_orders(self):
        blobs = {}
        for name, path in self.paths.items():
            with open(path, "rb") as fp:
                blobs[name] = fp.read()
        canonical = blobs["canonical"]
        for name, blob in blobs.items():
            self.assertEqual(
                blob, canonical,
                msg=f"corpus={CORPUS_TAG} order={name!r}: on-disk snapshot "
                    f"bytes differ from canonical",
            )

    def test_output_is_compact_sorted_json_with_one_newline(self):
        code, out, _ = self._search(
            self.paths["canonical"], "cat* OR fox*")
        self.assertEqual(code, 0)
        ids = json.loads(out)
        self.assertEqual(ids, sorted(ids))          # sorted by code point
        self.assertNotIn(", ", out)                 # compact separators
        self.assertNotIn(": ", out)
        self.assertTrue(out.endswith("\n"))
        self.assertEqual(len(out.splitlines()), 1)  # exactly one line


class InvalidQueryTests(DifferentialFixture):
    """Malformed queries: DataError at the module boundary; exit 2 / empty
    stdout / "error:" stderr at the CLI."""

    def test_enumeration_is_deterministic_and_covers_all_families(self):
        self.assertGreater(len(INVALID_QUERIES), 50)
        self.assertEqual({kind for kind, _ in INVALID_QUERIES}, {
            "missing-operand", "adjacent-operands", "empty-parentheses",
            "unbalanced-parentheses", "unterminated-quote", "illegal-star",
            "no-lexical-unit",
        })

    def test_module_entry_raises_exactly_data_error(self):
        snap = self.snapshots["canonical"]
        for family, raw in INVALID_QUERIES:
            with self.subTest(family=family, query=raw):
                with self.assertRaises(DataError) as ctx:
                    node = query.parse(raw)
                    if node is not None:
                        query.evaluate(node, snap)
                self.assertIs(type(ctx.exception), DataError)

    def test_cli_entry_exit_2_empty_stdout_error_prefix(self):
        snap_path = self._build(INPUT_ORDERS[0][1], "snap-a.json")
        for family, raw in INVALID_QUERIES:
            code, out, err = self._run(snap_path, raw)
            self.assertEqual(
                code, 2,
                msg=(f"corpus={CORPUS_TAG} family={family} query={raw!r}: "
                     f"expected exit 2, got {code}"),
            )
            self.assertEqual(
                out, "",
                msg=(f"corpus={CORPUS_TAG} family={family} query={raw!r}: "
                     f"stdout must be empty, got {out!r}"),
            )
            first = err.splitlines()[0] if err else ""
            self.assertTrue(
                first.startswith("error:"),
                msg=(f"corpus={CORPUS_TAG} family={family} query={raw!r}: "
                     f"first stderr line must start with 'error:', "
                     f"got {first!r}"),
            )

    def test_same_against_a_snapshot_from_another_input_order(self):
        snap_path = self._build(INPUT_ORDERS[1][1], "snap-b.json")
        for family, raw in INVALID_QUERIES:
            code, out, err = self._run(snap_path, raw)
            self.assertEqual(code, 2, msg=f"{family}: {raw!r}")
            self.assertEqual(out, "", msg=raw)
            self.assertTrue(err.splitlines()[0].startswith("error:"),
                            msg=raw)

    # -- helpers ------------------------------------------------------------
    def _build(self, docs, filename):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        docs_path = os.path.join(tmp.name, "docs.jsonl")
        snap_path = os.path.join(tmp.name, filename)
        with open(docs_path, "w", encoding="utf-8") as fp:
            fp.writelines(_jsonl(docs))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["build", docs_path, snap_path])
        self.assertEqual(code, 0, msg=err.getvalue())
        return snap_path

    @staticmethod
    def _run(snap_path, raw):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["search", snap_path, raw])
        return code, out.getvalue(), err.getvalue()


class ReferenceModelSelfTests(unittest.TestCase):
    """Lock the independent oracle's answers to hand-computed match sets."""

    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceModel(DOCS)

    def test_universe_contains_the_empty_text_documents(self):
        self.assertEqual(set(self.ref.universe),
                         {f"d0{i}" for i in range(1, 9)})

    def test_a_repeated_word_matches_its_document_once(self):
        self.assertEqual(set(self.ref.term_docs("fox")),
                         {"d03", "d04", "d05", "d08"})

    def test_accent_divergence_in_the_cafe_family(self):
        self.assertEqual(set(self.ref.term_docs("cafe")), {"d05"})
        self.assertEqual(set(self.ref.term_docs("café")), {"d04", "d07"})

    def test_phrases_require_consecutive_positions(self):
        # d04 has "quick fox" at positions (1,2); d05 and d08 too.
        self.assertEqual(set(self.ref.phrase_docs(("quick", "fox"))),
                         {"d04", "d05", "d08"})
        self.assertEqual(set(self.ref.phrase_docs(("fox", "fox"))), {"d03"})
        self.assertEqual(set(self.ref.phrase_docs(("file", "12"))), {"d04"})
        self.assertEqual(set(self.ref.phrase_docs(("quick", "fox", "jump"))),
                         set())
        self.assertEqual(set(self.ref.phrase_docs(("fox", "cat"))), set())

    def test_prefixes_are_codepoint_prefixes_on_normalized_terms(self):
        self.assertEqual(set(self.ref.prefix_docs("cat")),
                         {"d04", "d06", "d07"})
        self.assertEqual(set(self.ref.prefix_docs("ca")),
                         {"d04", "d05", "d06", "d07"})
        self.assertEqual(set(self.ref.prefix_docs("caf")),
                         {"d04", "d05", "d07"})
        self.assertEqual(set(self.ref.prefix_docs("cafe")), {"d05"})
        self.assertEqual(set(self.ref.prefix_docs("狸")), set())
        self.assertEqual(set(self.ref.prefix_docs("zzz")), set())

    def test_not_universe_is_all_documents(self):
        not_fox = self.ref.eval_tree(("not", ("term", "fox")))
        self.assertEqual(set(not_fox),
                         set(self.ref.universe)
                         - set(self.ref.term_docs("fox")))
        self.assertIn("d01", not_fox)
        self.assertIn("d02", not_fox)

    def test_precedence_not_and_or(self):
        t = self.ref
        a, b, c = ("term", "fox"), ("term", "cat"), ("term", "jump")
        self.assertEqual(
            t.eval_tree(("or", a, ("and", b, c))),
            t.term_docs("fox") | (t.term_docs("cat") & t.term_docs("jump")),
        )
        self.assertEqual(
            t.eval_tree(("and", ("not", a), b)),
            (set(t.universe) - t.term_docs("fox")) & t.term_docs("cat"),
        )


if __name__ == "__main__":
    unittest.main()
