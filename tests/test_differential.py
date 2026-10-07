"""Deterministic differential tests for the query parser and set evaluator.

These tests never hand-compute expected hits from the production code.
Instead a small *reference semantics* is implemented in this file directly on
top of the raw documents: it NFKC-normalizes/casefolds and tokenizes with the
standard ``unicodedata`` module (it deliberately does **not** import the
package's ``analysis.tokenize``), then derives match sets from the normalized
word order itself:

* a term matches a document iff the normalized word occurs in it;
* a phrase requires consecutive positions, checked against word indices;
* a prefix unions every normalized dictionary term it is a codepoint prefix
  of (plain ``str.startswith``, no collation);
* NOT's universe is the set of *every* document, including empty-text ones;
* NOT > AND > OR is expressed by the expected expression tree, which is built
  at the same time as its surface query string.

Everything here is fixed data: no randomness, no timestamps, no locale, no
third-party libraries, no reliance on set/dict iteration order feeding an
assertion (sets are always sorted before comparison), so a run produces the
same corpus, query list and assertion order on every machine.

Public behavior (module API, CLI, snapshot version, existing fixed samples) is
only read, never changed.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from text_index_engine import query
from text_index_engine.__main__ import main
from text_index_engine.errors import DataError
from text_index_engine.snapshot import Snapshot, build_from_lines

# ---------------------------------------------------------------------------
# Fixed controlled corpus: empty text, separator-only text, repeated words,
# NFKC-compatible characters (fullwidth/ligature), case variants, documents
# that share prefixes, CJK shared prefixes, lowercase keyword words and
# letter/digit runs (digits do not split a token).
# ---------------------------------------------------------------------------
CORPUS = [
    {"id": "d01", "text": ""},
    {"id": "d02", "text": "   ...  !!!  "},
    {"id": "d03", "text": "alpha alpha beta"},
    {"id": "d04", "text": "Beta GAMMA gamma delta"},
    {"id": "d05", "text": "apple application ape band"},
    {"id": "d06", "text": "café ｃａｆé ﬁle"},
    {"id": "d07", "text": "CAFÉ Cafe"},
    {"id": "d08", "text": "alpha beta gamma"},
    {"id": "d09", "text": "gamma gamma"},
    {"id": "d10", "text": "快狐 快狐 跳过"},
    {"id": "d11", "text": "狐狸 快速"},
    {"id": "d12", "text": "and or not AND"},
    {"id": "d13", "text": "item1 item12 item2"},
]
CORPUS_TAG = "diff-corpus"

# Four fixed input orders of the same documents.
_ORDER0 = list(range(len(CORPUS)))
ORDER_INDEX = [
    _ORDER0,
    _ORDER0[::-1],
    _ORDER0[4:] + _ORDER0[:4],
    [12, 0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6],
]
ORDER_NAMES = ["order0-original", "order1-reversed",
               "order2-rotate4", "order3-interleave"]


def _docs_for(order):
    return [CORPUS[i] for i in order]


def _jsonl(docs):
    return [json.dumps(d, ensure_ascii=False) + "\n" for d in docs]


# ---------------------------------------------------------------------------
# Independent reference model (must not call any production query/analysis
# function to derive an expectation).
# ---------------------------------------------------------------------------
def ref_tokenize(text):
    """NFKC + casefold then maximal L*/N* runs, straight from raw text."""
    tokens = []
    run = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        if unicodedata.category(ch)[0] in ("L", "N"):
            run.append(ch)
        elif run:
            tokens.append("".join(run))
            run = []
    if run:
        tokens.append("".join(run))
    return tokens


def build_reference(docs):
    """words per document, universe, postings (term -> doc -> positions),
    and the codepoint-sorted normalized dictionary."""
    words = {d["id"]: ref_tokenize(d["text"]) for d in docs}
    universe = frozenset(words)
    postings = {}
    for doc_id, toks in words.items():
        for pos, term in enumerate(toks):
            postings.setdefault(term, {}).setdefault(doc_id, set()).add(pos)
    return {
        "words": words,
        "universe": universe,
        "postings": postings,
        "dictionary": sorted(postings),
    }


def _phrase_hits(postings, phrase_words):
    first = postings.get(phrase_words[0])
    if not first:
        return set()
    hits = set()
    for doc_id, starts in first.items():
        later = [postings.get(w, {}).get(doc_id, frozenset())
                 for w in phrase_words[1:]]
        for start in starts:
            if all(start + k in later[k - 1]
                   for k in range(1, len(phrase_words))):
                hits.add(doc_id)
                break
    return hits


def ref_atom(model, kind, raw):
    postings = model["postings"]
    if kind == "word":
        term = ref_tokenize(raw)[0]
        return set(postings.get(term, ()))
    if kind == "prefix":
        prefix = ref_tokenize(raw[:-1])[0]
        hits = set()
        # Codepoint-by-codepoint prefix over normalized dictionary entries.
        for term in model["dictionary"]:
            if term.startswith(prefix):
                hits.update(postings[term])
        return hits
    if kind == "phrase":
        return _phrase_hits(postings, ref_tokenize(raw[1:-1]))
    raise AssertionError(f"unknown reference atom kind {kind!r}")


def ref_evaluate(model, node):
    kind = node[0]
    if kind == "atom":
        return ref_atom(model, node[1], node[2])
    if kind == "paren":
        return ref_evaluate(model, node[1])
    if kind == "not":
        return set(model["universe"]) - ref_evaluate(model, node[1])
    if kind == "and":
        return ref_evaluate(model, node[1]) & ref_evaluate(model, node[2])
    if kind == "or":
        return ref_evaluate(model, node[1]) | ref_evaluate(model, node[2])
    raise AssertionError(f"unknown reference node {kind!r}")


# ---------------------------------------------------------------------------
# Leaf query specs: (raw query, atom kind). Expected sets come from the
# reference model, never from production.
# ---------------------------------------------------------------------------
CASE_VARIANT_WORDS = ["FILE", "ＧＡＭＭＡ"]
ABSENT_WORDS = ["zzz", "ALPHAZ", "42", "①", "caf", "快x", "item20", "item"]
PHRASES = [
    '"alpha beta"', '"ALPHA BETA"', '"beta gamma"', '"alpha alpha"',
    '"gamma gamma"', '"gamma delta"', '"apple band"', '"alpha gamma"',
    '"快狐 快狐"', '"快狐 跳过"', '"café ﬁle"', '"alpha beta gamma"',
    '"gamma"', '"alpha, beta"', '"zzz alpha"', '"ｃａｆé"',
    '"or not"', '"not AND"', '"item1 item12"', '"item12 item2"',
    '"café café"', '"quick brown"',
]
PREFIXES = [
    "al*", "alph*", "app*", "ap*", "a*", "ca*", "caf*", "CAF*", "ﬁ*",
    "b*", "g*", "ga*", "band*", "de*", "快*", "狐*", "z*", "apple*",
    "applex*", "deltaz*", "①*", "an*", "i*", "item*", "item1*",
    "item2*", "item12*", "ITEM*", "itemz*",
]

# Pool systematically fed to the boolean combinators; it deliberately mixes
# terms, phrases (matching and non-consecutive) and prefixes (matching and
# absent), ASCII and non-ASCII.
POOL_SPECS = [
    ("alpha", "word"),
    ("band", "word"),
    ("gamma", "word"),
    ("delta", "word"),
    ("zzz", "word"),
    ("café", "word"),
    ("快狐", "word"),
    ("and", "word"),
    ('"alpha beta"', "phrase"),
    ('"apple band"', "phrase"),
    ('"gamma gamma"', "phrase"),
    ("app*", "prefix"),
    ("b*", "prefix"),
    ("z*", "prefix"),
    ("item1*", "prefix"),
    ("快*", "prefix"),
]


def generate_legal(model):
    """Return a deterministically ordered list of (raw query, ref node)."""
    items = {}

    def add(raw, node):
        items.setdefault(raw, node)

    # Every normalized dictionary term as a plain word, plus case variants and
    # words guaranteed absent from the dictionary.
    for term in model["dictionary"]:
        add(term, ("atom", "word", term))
    for raw in CASE_VARIANT_WORDS + ABSENT_WORDS:
        add(raw, ("atom", "word", raw))
    for raw in PHRASES:
        add(raw, ("atom", "phrase", raw))
    for raw in PREFIXES:
        add(raw, ("atom", "prefix", raw))

    pool = [(raw, ("atom", kind, raw)) for raw, kind in POOL_SPECS]

    # Layer 1: parenthesized atoms, NOT atoms, all ordered AND/OR pairs.
    for raw, node in pool:
        add(f"({raw})", ("paren", node))
        add(f"NOT {raw}", ("not", node))
    for raw_a, node_a in pool:
        for raw_b, node_b in pool:
            add(f"{raw_a} AND {raw_b}", ("and", node_a, node_b))
            add(f"{raw_a} OR {raw_b}", ("or", node_a, node_b))

    # Layer 2: double NOT, NOT over grouped binaries, and fixed templates that
    # force NOT/AND/OR precedence and left/right grouping to differ.
    for raw, node in pool:
        add(f"NOT NOT {raw}", ("not", ("not", node)))
    for raw_a, node_a in pool:
        for raw_b, node_b in pool:
            add(f"NOT ({raw_a} AND {raw_b})",
                ("not", ("and", node_a, node_b)))
            add(f"NOT ({raw_a} OR {raw_b})",
                ("not", ("or", node_a, node_b)))
    n = len(pool)
    for i, (raw_a, node_a) in enumerate(pool):
        raw_b, node_b = pool[(i + 6) % n]
        raw_c, node_c = pool[(i + 11) % n]
        add(f"NOT {raw_a} OR {raw_b}",
            ("or", ("not", node_a), node_b))
        add(f"{raw_a} OR {raw_b} AND {raw_c}",
            ("or", node_a, ("and", node_b, node_c)))
        add(f"({raw_a} AND {raw_b}) OR {raw_c}",
            ("or", ("and", node_a, node_b), node_c))
        add(f"{raw_a} AND ({raw_b} OR {raw_c})",
            ("and", node_a, ("or", node_b, node_c)))
        add(f"NOT ({raw_a} OR {raw_b}) AND {raw_c}",
            ("and", ("not", ("or", node_a, node_b)), node_c))
        add(f"NOT NOT {raw_a} AND {raw_b}",
            ("and", ("not", ("not", node_a)), node_b))
        add(f"({raw_a}) OR NOT ({raw_b} AND {raw_c})",
            ("or", ("paren", node_a),
             ("not", ("paren", ("and", node_b, node_c)))))

    return list(items.items())


def generate_invalid():
    """Fixed + deterministically generated malformed queries.

    Categories: missing operands, adjacent operands, empty parentheses,
    unterminated quotes, illegal star positions, unbalanced parentheses,
    phrases with no searchable word, and bare runs with no lexical unit.
    """
    q = [
        # Missing operands.
        "AND", "OR", "NOT", "AND alpha", "OR alpha", "alpha AND",
        "alpha OR", "NOT AND", "alpha AND NOT", "(alpha OR)",
        "alpha AND (OR beta)",
        # Empty parentheses.
        "()", "(())", "NOT ()", "alpha AND ()", "() AND alpha",
        # Phrases with no searchable word.
        '"!!!"', '"..."', '"*"',
        # Bare runs carrying no lexical unit / several terms without operator.
        "...", "!!!", "alpha,beta", "café,band",
        # Illegal star positions.
        "*", "**", "alpha**", "*alpha", "alpha*beta", "al*ph*", "a*b*",
        "alpha *", "* alpha", "!!!*", "alpha,beta*", "(alpha)*",
        "foo**", "fo*o*",
        # Unbalanced parentheses.
        "(alpha", "alpha)", "((alpha)", "(alpha))", "((alpha AND beta)",
        "alpha AND (beta OR gamma", "(alpha AND beta))", ")(",
        "NOT (alpha AND beta", "((alpha)))",
        # Unterminated quotes.
        '"', '"alpha', 'alpha "beta gamma', '"café', 'NOT "alpha',
        '"alpha beta',
    ]

    # Adjacent operands generated deterministically from the token space:
    # every ordered pair of a fixed leaf slice, including phrases/prefixes.
    adj_leaves = [raw for raw, _ in POOL_SPECS]
    adj_slice = [adj_leaves[i] for i in (0, 1, 2, 4, 6, 8, 11, 14)]
    q.extend(f"{a} {b}" for a in adj_slice for b in adj_slice)
    q.extend([
        "(alpha) beta", "alpha (beta)", '"alpha beta" gamma',
        'gamma "alpha beta"', "app* band", "band app*",
    ])

    # Illegal star placements generated over normalized dictionary words.
    for term in ("alpha", "café", "item1", "快狐"):
        q.extend([f"*{term}", f"{term}**", f"{term} *"])
    q.append("alpha*café")
    q.append("item1*item2")

    # Unterminated quotes generated from phrase-shaped contents.
    for content in ("alpha", "alpha beta", "café", "快狐 跳过", "gamma, delta"):
        q.append(f'"{content}')

    # De-duplicate while preserving the fixed generation order.
    seen = set()
    unique = []
    for raw in q:
        if raw not in seen:
            seen.add(raw)
            unique.append(raw)
    return unique


MODEL = build_reference(CORPUS)
LEGAL_QUERIES = generate_legal(MODEL)
INVALID_QUERIES = generate_invalid()
SNAPSHOT_BYTES = [build_from_lines(_jsonl(_docs_for(order)))
                  for order in ORDER_INDEX]


def expected_line(hits):
    """The unique CLI representation: codepoint-sorted compact JSON + LF."""
    return json.dumps(sorted(hits), ensure_ascii=False,
                      separators=(",", ":")) + "\n"


def _detail(corpus_id, raw, expected, actual):
    return (f"[corpus={CORPUS_TAG}/{corpus_id}] query={raw!r} "
            f"expected={expected!r} actual={actual!r}")


class DifferentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.snap_paths = []
        for k, data in enumerate(SNAPSHOT_BYTES):
            path = os.path.join(self.dir, f"snapshot{k}.json")
            with open(path, "wb") as fp:
                fp.write(data)
            self.snap_paths.append(path)
        self.docs_path = os.path.join(self.dir, "docs.jsonl")
        with open(self.docs_path, "w", encoding="utf-8") as fp:
            fp.writelines(_jsonl(CORPUS))
        self.snapshots = [Snapshot.load(data) for data in SNAPSHOT_BYTES]

    def tearDown(self):
        self.tmp.cleanup()

    def _cli_search(self, snap_path, raw):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["search", snap_path, raw])
        return code, out.getvalue(), err.getvalue()

    # -- corpus/query generation itself ------------------------------------
    def test_generation_is_deterministic_and_sized(self):
        again = generate_legal(build_reference(CORPUS))
        self.assertEqual(
            [q for q, _ in again], [q for q, _ in LEGAL_QUERIES])
        self.assertEqual(generate_invalid(), INVALID_QUERIES)
        # The combinatorial sweep is intentionally large.
        self.assertGreater(len(LEGAL_QUERIES), 1000)
        self.assertGreater(len(INVALID_QUERIES), 100)
        digest = hashlib.sha256(
            "\x1f".join(q for q, _ in LEGAL_QUERIES).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            digest,
            hashlib.sha256(
                "\x1f".join(q for q, _ in again).encode("utf-8")
            ).hexdigest(),
        )

    # -- snapshot determinism across input orders --------------------------
    def test_snapshot_bytes_identical_for_all_input_orders(self):
        canonical = SNAPSHOT_BYTES[0]
        for k, data in enumerate(SNAPSHOT_BYTES):
            self.assertEqual(
                data, canonical,
                f"[corpus={CORPUS_TAG}/{ORDER_NAMES[k]}] snapshot bytes "
                f"differ from {ORDER_NAMES[0]}",
            )

    # -- module-level differential: every legal query, canonical snapshot --
    def test_legal_queries_match_reference_module_sets(self):
        snap = self.snapshots[0]
        for raw, node in LEGAL_QUERIES:
            expected = sorted(ref_evaluate(MODEL, node))
            try:
                actual = sorted(query.evaluate(query.parse(raw), snap))
            except Exception as exc:  # noqa: BLE001 - report with full detail
                self.fail(
                    _detail(ORDER_NAMES[0], raw, expected,
                            f"raised {type(exc).__name__}: {exc}"))
            self.assertEqual(
                actual, expected,
                _detail(ORDER_NAMES[0], raw, expected, actual))

    # -- CLI differential: every legal query, canonical snapshot -----------
    def test_legal_queries_match_reference_cli_output(self):
        for raw, node in LEGAL_QUERIES:
            expected = expected_line(ref_evaluate(MODEL, node))
            code, out, err = self._cli_search(self.snap_paths[0], raw)
            self.assertEqual(
                code, 0,
                _detail(ORDER_NAMES[0], raw, expected,
                        f"exit={code} stderr={err!r}"))
            self.assertEqual(
                out, expected,
                _detail(ORDER_NAMES[0], raw, expected, out))
            # The representation is exactly one compact JSON array plus LF.
            self.assertTrue(out.endswith("\n") and out.count("\n") == 1,
                            _detail(ORDER_NAMES[0], raw, expected, out))
            self.assertEqual(json.loads(out), json.loads(expected))

    # -- all orders: same module sets and byte-identical CLI output --------
    def test_all_orders_give_byte_identical_query_output(self):
        expected_sets = [ref_evaluate(MODEL, node)
                         for _, node in LEGAL_QUERIES]
        expected_lines = [expected_line(hits) for hits in expected_sets]
        base_bytes = [line.encode("utf-8") for line in expected_lines]

        for k in range(1, len(ORDER_INDEX)):
            snap = self.snapshots[k]
            cid = ORDER_NAMES[k]
            for idx, (raw, node) in enumerate(LEGAL_QUERIES):
                module_hits = sorted(
                    query.evaluate(query.parse(raw), snap))
                self.assertEqual(
                    module_hits, sorted(expected_sets[idx]),
                    _detail(cid, raw, sorted(expected_sets[idx]),
                            module_hits))
                code, out, err = self._cli_search(
                    self.snap_paths[k], raw)
                self.assertEqual(
                    code, 0,
                    _detail(cid, raw, expected_lines[idx],
                            f"exit={code} stderr={err!r}"))
                self.assertEqual(
                    out.encode("utf-8"), base_bytes[idx],
                    _detail(cid, raw, base_bytes[idx],
                            out.encode("utf-8")))

    # -- invalid queries: exactly DataError at the module entry ------------
    def test_invalid_queries_raise_exactly_data_error(self):
        snap = self.snapshots[0]
        for raw in INVALID_QUERIES:
            try:
                node = query.parse(raw)
            except DataError as exc:
                self.assertIs(
                    type(exc), DataError,
                    _detail(ORDER_NAMES[0], raw, "DataError",
                            f"subclass {type(exc).__name__}"))
                continue
            try:
                if node is not None:
                    query.evaluate(node, snap)
            except DataError as exc:
                self.assertIs(type(exc), DataError)
                continue
            self.fail(
                _detail(ORDER_NAMES[0], raw, "DataError",
                        f"parsed/evaluated to {node!r}"))

    # -- invalid queries: exit 2, empty stdout, error: on stderr -----------
    def test_invalid_queries_cli_exit_2_empty_stdout(self):
        for raw in INVALID_QUERIES:
            code, out, err = self._cli_search(
                self.snap_paths[0], raw)
            self.assertEqual(
                code, 2,
                _detail(ORDER_NAMES[0], raw, "exit 2",
                        f"exit {code} stdout={out!r}"))
            self.assertEqual(
                out, "",
                _detail(ORDER_NAMES[0], raw, "empty stdout", repr(out)))
            first = err.splitlines()[0] if err.splitlines() else ""
            self.assertTrue(
                first.startswith("error:"),
                _detail(ORDER_NAMES[0], raw,
                        "stderr first line 'error: ...'", repr(first)))

    # -- true process: raw stdout bytes and OS-level exit code -------------
    def test_process_level_bytes_and_exit_codes(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get(
            "PYTHONPATH", "")

        # CLI build of one order must emit the same canonical bytes.
        built = os.path.join(self.dir, "built.json")
        proc = subprocess.run(
            [sys.executable, "-m", "text_index_engine", "build",
             self.docs_path, built],
            cwd=repo_root, env=env, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(built, "rb") as fp:
            self.assertEqual(fp.read(), SNAPSHOT_BYTES[0])

        # Fixed positions into the deterministically generated query list, so
        # membership is guaranteed without hand-picking surface strings.
        legal_subset = [
            LEGAL_QUERIES[i]
            for i in (0, 25, 77, 250, 650, len(LEGAL_QUERIES) - 1)
        ]
        for raw, node in legal_subset:
            proc = subprocess.run(
                [sys.executable, "-m", "text_index_engine", "search",
                 built, raw],
                cwd=repo_root, env=env, capture_output=True, check=False)
            expected = expected_line(ref_evaluate(MODEL, node)).encode("utf-8")
            self.assertEqual(proc.returncode, 0,
                             _detail("process", raw, 0, proc.returncode))
            self.assertEqual(
                proc.stdout, expected,
                _detail("process", raw, expected, proc.stdout))
            self.assertEqual(proc.stderr, b"")

        for raw in ("*", "alpha AND", '"unterminated', "()", "alpha)"):
            proc = subprocess.run(
                [sys.executable, "-m", "text_index_engine", "search",
                 built, raw],
                cwd=repo_root, env=env, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 2,
                             _detail("process", raw, 2, proc.returncode))
            self.assertEqual(proc.stdout, b"",
                             _detail("process", raw, b"", proc.stdout))
            first = proc.stderr.splitlines()[0] if proc.stderr.splitlines() \
                else b""
            self.assertTrue(
                first.startswith(b"error:"),
                _detail("process", raw, "error: line", first))


if __name__ == "__main__":
    unittest.main()
