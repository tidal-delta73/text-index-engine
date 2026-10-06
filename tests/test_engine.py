"""End-to-end and unit tests for the build/search vertical slice."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from text_index_engine import query
from text_index_engine.analysis import tokenize
from text_index_engine.__main__ import main
from text_index_engine.errors import DataError
from text_index_engine.snapshot import Snapshot, build_from_lines

DOCS = [
    {"id": "a", "text": "The quick brown fox jumps over the lazy dog"},
    {"id": "b", "text": "A quick fox and a quick dog"},
    {"id": "c", "text": ""},
    {"id": "d", "text": "café CAFÉ ﬁle"},
    {"id": "中", "text": "快狐 跳过 懒狗 quick"},
]


def lines(docs):
    return [json.dumps(d, ensure_ascii=False) + "\n" for d in docs]


class AnalysisTests(unittest.TestCase):
    def test_casefold_and_token_runs(self):
        self.assertEqual(
            tokenize("The  quick-brown_fox!"),
            ["the", "quick", "brown", "fox"],
        )

    def test_nfkc_ligature_and_accents(self):
        self.assertEqual(tokenize("ﬁle"), ["file"])
        self.assertEqual(tokenize("ｃａｆé"), ["café"])
        self.assertEqual(tokenize("①２"), ["12"])  # NFKC expands, then one run

    def test_unicode_letters(self):
        self.assertEqual(tokenize("快狐 跳过"), ["快狐", "跳过"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("  - ! "), [])


class BuildTests(unittest.TestCase):
    def test_canonical_shape_and_order(self):
        obj = json.loads(build_from_lines(lines(DOCS)))
        self.assertEqual(obj["version"], 1)
        # Documents sorted by code point ("中" U+4E2D after ASCII).
        self.assertEqual(obj["documents"], ["a", "b", "c", "d", "中"])
        quick = next(t for t in obj["terms"] if t["term"] == "quick")
        self.assertEqual(
            quick["postings"],
            [
                {"id": "a", "positions": [1]},
                {"id": "b", "positions": [1, 5]},
                {"id": "中", "positions": [3]},
            ],
        )

    def test_byte_identical_regardless_of_line_order(self):
        first = build_from_lines(lines(DOCS))
        shuffled = lines(list(reversed(DOCS)))
        self.assertEqual(first, build_from_lines(shuffled))
        # Shuffle differently: rotate as well.
        rotated = lines(DOCS[2:] + DOCS[:2])
        self.assertEqual(first, build_from_lines(rotated))

    def test_no_environment_data(self):
        raw = build_from_lines(lines(DOCS)).decode("utf-8")
        self.assertNotIn(tempfile.gettempdir(), raw)
        self.assertNotIn(os.getcwd(), raw)

    def assert_data_error(self, doc_lines, fragment=None):
        with self.assertRaises(DataError) as ctx:
            build_from_lines(doc_lines)
        if fragment:
            self.assertIn(fragment, str(ctx.exception))

    def test_invalid_lines(self):
        good = json.dumps({"id": "x", "text": "hello"}) + "\n"
        self.assert_data_error(["not json\n"], "not a JSON")
        self.assert_data_error(["[1,2]\n"], "object")
        self.assert_data_error(['{"id":"x"}\n'], "text")
        self.assert_data_error(['{"text":"y"}\n'], "id")
        self.assert_data_error(['{"id":1,"text":"y"}\n'], "string")
        self.assert_data_error(['{"id":"x","text":2}\n'], "string")
        self.assert_data_error(['{"id":"","text":"y"}\n'], "non-empty")
        self.assert_data_error([good, good], "duplicate")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(lines(DOCS)))

    def run_query(self, q):
        return sorted(query.evaluate(query.parse(q), self.snap))

    def test_basic_terms_and_normalization(self):
        self.assertEqual(self.run_query("FOX"), ["a", "b"])  # casefold
        self.assertEqual(self.run_query("ﬁle"), ["d"])       # NFKC ligature
        self.assertEqual(self.run_query("missing"), [])

    def test_phrase_requires_consecutive_positions(self):
        self.assertEqual(self.run_query('"quick fox"'), ["b"])
        self.assertEqual(self.run_query('"quick brown fox"'), ["a"])
        self.assertEqual(self.run_query('"fox dog"'), [])   # not consecutive

    def test_precedence_not_and_or(self):
        # a,b contain fox; b contains dog... set up explicit expectations:
        fox = {"a", "b"}
        dog = {"a", "b"}
        not_fox = {"c", "d", "中"}
        # OR binds looser: "fox OR quick AND dog" == fox OR (quick AND dog)
        self.assertEqual(set(self.run_query("fox OR quick AND dog")),
                         fox | ({"a", "b", "中"} & dog))
        self.assertEqual(set(self.run_query("NOT fox")), not_fox)
        self.assertEqual(set(self.run_query("NOT fox AND quick")),
                         not_fox & {"a", "b", "中"})
        self.assertEqual(set(self.run_query("NOT (fox OR quick)")),
                         {"c", "d"})

    def test_not_universe_includes_empty_text_docs(self):
        self.assertIn("c", set(self.run_query("NOT fox")))

    def test_parentheses(self):
        self.assertEqual(
            self.run_query("(quick OR café) AND fox"),
            ["a", "b"],
        )

    def test_lowercase_operators_are_terms(self):
        # "and" is an ordinary word in document b.
        self.assertEqual(self.run_query("and"), ["b"])

    def test_empty_query(self):
        self.assertIsNone(query.parse(""))

    def assert_rejected(self, q):
        with self.assertRaises(DataError):
            node = query.parse(q)
            query.evaluate(node, self.snap)

    def test_malformed_queries(self):
        self.assert_rejected("AND fox")
        self.assert_rejected("fox AND")
        self.assert_rejected("NOT")
        self.assert_rejected("fox OR")
        self.assert_rejected("(fox OR dog")
        self.assert_rejected("fox OR dog)")
        self.assert_rejected('"unterminated')
        self.assert_rejected('"!!!"')            # phrase without terms
        self.assert_rejected("...")              # no lexical unit
        self.assert_rejected("fox,dog")          # no implicit operator
        self.assert_rejected("fox dog")          # no implicit operator
        self.assert_rejected("fox AND (dog OR)")


# Separate documents for prefix semantics: several terms share stems, and
# "cafe" vs "café" differ only past a common ASCII prefix (code-point prefix
# matching must not treat them as equal).
PREFIX_DOCS = [
    {"id": "p1", "text": "apple application app"},
    {"id": "p2", "text": "APRICOT banana apply"},
    {"id": "p3", "text": "café"},
    {"id": "p4", "text": "cafe"},
    {"id": "p5", "text": "应用 苹果"},
    {"id": "p6", "text": ""},
]


class PrefixSearchTests(unittest.TestCase):
    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(lines(PREFIX_DOCS)))

    def run_query(self, q):
        return sorted(query.evaluate(query.parse(q), self.snap))

    def test_ast_shape(self):
        self.assertEqual(query.parse("app*"), ("prefix", "app"))
        # The stored value is the normalized body without the star.
        self.assertEqual(query.parse("APP*"), ("prefix", "app"))

    def test_prefix_unions_all_matching_terms(self):
        # p1 holds three different app*-terms but appears once.
        self.assertEqual(self.run_query("app*"), ["p1", "p2"])
        self.assertEqual(self.run_query("ap*"), ["p1", "p2"])
        self.assertEqual(self.run_query("appl*"), ["p1", "p2"])
        self.assertEqual(self.run_query("apple*"), ["p1"])
        self.assertEqual(self.run_query("ban*"), ["p2"])
        self.assertEqual(self.run_query("apricot*"), ["p2"])

    def test_prefix_includes_exact_term(self):
        self.assertEqual(self.run_query("apple*"), ["p1"])
        self.assertEqual(self.run_query("apple"), ["p1"])

    def test_casefold_prefix(self):
        self.assertEqual(self.run_query("APP*"), ["p1", "p2"])
        self.assertEqual(self.run_query("Apricot*"), ["p2"])

    def test_nfkc_prefix(self):
        # Fullwidth letters normalize before the prefix is taken.
        self.assertEqual(self.run_query("ａｐ*"), ["p1", "p2"])

    def test_unicode_codepoint_prefix(self):
        self.assertEqual(self.run_query("应*"), ["p5"])
        self.assertEqual(self.run_query("苹*"), ["p5"])
        # Pure code-point matching: "cafe*" must not reach "café", and the
        # accented spelling is its own prefix. Shared ASCII stem matches both.
        self.assertEqual(self.run_query("cafe*"), ["p4"])
        self.assertEqual(self.run_query("café*"), ["p3"])
        self.assertEqual(self.run_query("caf*"), ["p3", "p4"])

    def test_no_match_is_empty_set(self):
        self.assertEqual(self.run_query("zzz*"), [])
        self.assertEqual(self.run_query("applex*"), [])
        # No stemming or locale folding: "café*" never matches "cafe".
        self.assertEqual(self.run_query("应用x*"), [])

    def test_boolean_combinations(self):
        # app* = {p1,p2}; apple = {p1}
        self.assertEqual(self.run_query("app* AND NOT apple"), ["p2"])
        self.assertEqual(self.run_query("banana OR app*"), ["p1", "p2"])
        self.assertEqual(self.run_query("(app* OR café*) AND banana"),
                         ["p2"])
        # caf* = {p3,p4}, café* = {p3}
        self.assertEqual(self.run_query("caf* AND NOT café*"), ["p4"])
        self.assertEqual(self.run_query("NOT app*"), ["p3", "p4", "p5", "p6"])
        # A prefix matching nothing leaves NOT's whole universe intact.
        self.assertEqual(
            self.run_query("NOT zzz*"), ["p1", "p2", "p3", "p4", "p5", "p6"])
        self.assertEqual(self.run_query("app* OR zzz*"), ["p1", "p2"])
        self.assertEqual(self.run_query("app* AND zzz*"), [])
        self.assertEqual(self.run_query("NOT NOT app*"), ["p1", "p2"])

    def test_parenthesized_prefix(self):
        self.assertEqual(self.run_query("(app*)"), ["p1", "p2"])
        self.assertEqual(self.run_query("((ap*)) AND banana"), ["p2"])

    def test_starred_keywords_are_prefix_terms(self):
        # With a star they are ordinary prefix operands, not operators.
        self.assertEqual(query.parse("AND*"), ("prefix", "and"))
        self.assertEqual(query.parse("OR*"), ("prefix", "or"))
        self.assertEqual(query.parse("NOT*"), ("prefix", "not"))
        # No "and"/"or"/"not" terms in this corpus, so they match nothing;
        # crucially this is an empty result, not an operator-usage error.
        self.assertEqual(self.run_query("AND*"), [])
        self.assertEqual(self.run_query("NOT*"), [])
        # The same words without a star stay operators/terms as before.
        snap = Snapshot.load(build_from_lines(lines(DOCS)))
        self.assertEqual(
            sorted(query.evaluate(query.parse("AND*"), snap)), ["b"])
        with self.assertRaises(DataError):
            query.parse("AND fox")

    def test_star_literal_inside_phrase(self):
        # The star gets no wildcard meaning: phrase analysis is unchanged,
        # and "quick*" analyzes to the single phrase word "quick".
        snap = Snapshot.load(build_from_lines(lines(DOCS)))
        self.assertEqual(
            sorted(query.evaluate(query.parse('"quick*"'), snap)),
            ["a", "b", "中"],
        )
        # In the prefix corpus no term "ap" exists, so the phrase matches
        # nothing rather than behaving like ap*.
        self.assertEqual(self.run_query('"ap*"'), [])

    def assert_rejected(self, q):
        with self.assertRaises(DataError) as ctx:
            node = query.parse(q)
            query.evaluate(node, self.snap)
        return str(ctx.exception)

    def test_illegal_star_forms_rejected(self):
        for q in (
            "*",          # bare star: empty body
            "**",         # multiple stars
            "***",
            "app**",      # multiple stars
            "*app",       # star not at the end
            "a*p",        # star in the middle
            "app*le",     # trailing text after star
            "a*b*",       # several stars
            "!*",         # body analyzes to zero tokens
            "-*",
            "a,b*",       # body analyzes to multiple tokens
            "(app**)",
            "(*)",
            "app *",      # separate bare star run
        ):
            msg = self.assert_rejected(q)
            self.assertTrue(msg.startswith("query:"), q)

    def test_prefix_adjacency_still_requires_operator(self):
        for q in (
            "app* banana",
            "apple app*",
            '"apple" app*',
            "app* (banana)",
            "(app*)banana",
            "cafe app*",
        ):
            self.assert_rejected(q)

    def test_prefix_normalization_matches_terms(self):
        # Query-side analysis uses the same rules as the index: a star after
        # text that NFKC-expands into one run is one prefix.
        self.assertEqual(self.run_query("café*"), ["p3"])
        # Punctuation in the body splits it and must therefore be rejected.
        self.assert_rejected("ap.ple*")


class PrefixDeterminismTests(unittest.TestCase):
    def _hits_json(self, raw_snapshot, q):
        snap = Snapshot.load(raw_snapshot)
        hits = sorted(query.evaluate(query.parse(q), snap))
        return json.dumps(hits, ensure_ascii=False, separators=(",", ":"))

    def test_equivalent_snapshots_same_prefix_results(self):
        orders = [
            PREFIX_DOCS,
            list(reversed(PREFIX_DOCS)),
            PREFIX_DOCS[3:] + PREFIX_DOCS[:3],
        ]
        snapshots = [build_from_lines(lines(order)) for order in orders]
        # The snapshots themselves are byte-identical v1 output.
        for raw in snapshots[1:]:
            self.assertEqual(raw, snapshots[0])
        for q in ("app*", "ap*", "caf*", "cafe*", "café*", "应*", "zzz*",
                  "app* AND NOT apple", "(caf* OR 苹*) AND NOT banana"):
            outputs = {self._hits_json(raw, q) for raw in snapshots}
            self.assertEqual(len(outputs), 1, (q, outputs))


class PrefixOldSnapshotTests(unittest.TestCase):
    def test_hand_built_v1_snapshot_supports_prefixes(self):
        # An already-existing v1 snapshot (no rebuild): the loader gains no
        # new requirements, and prefix queries work against its terms.
        raw = json.dumps({"version": 1, "documents": ["d1", "d2"], "terms": [
            {"term": "app",
             "postings": [{"id": "d1", "positions": [0]}]},
            {"term": "apple",
             "postings": [{"id": "d2", "positions": [0]}]},
        ]}, ensure_ascii=False)
        snap = Snapshot.load(raw)
        self.assertEqual(
            sorted(query.evaluate(query.parse("app*"), snap)),
            ["d1", "d2"],
        )
        self.assertEqual(
            sorted(query.evaluate(query.parse("apple*"), snap)), ["d2"])
        self.assertEqual(
            sorted(query.evaluate(query.parse("apx*"), snap)), [])


class SnapshotLoadTests(unittest.TestCase):
    def _obj(self, docs=DOCS):
        return json.loads(build_from_lines(lines(docs)))

    def assert_rejected(self, obj):
        with self.assertRaises(DataError):
            Snapshot.load(json.dumps(obj, ensure_ascii=False))

    def test_unsupported_version(self):
        obj = self._obj()
        obj["version"] = 999
        with self.assertRaises(DataError):
            Snapshot.load(json.dumps(obj))

    def test_garbage(self):
        with self.assertRaises(DataError):
            Snapshot.load(b"not json")
        with self.assertRaises(DataError):
            Snapshot.load(b"\xff\xfe")

    def test_built_snapshots_accepted(self):
        # The ordinary corpus, including an empty-text document.
        Snapshot.load(build_from_lines(lines(DOCS)))
        # Empty document set.
        Snapshot.load(build_from_lines([]))
        # A document set consisting only of empty-text documents.
        Snapshot.load(build_from_lines(lines([
            {"id": "x", "text": ""},
            {"id": "y", "text": "   - ! "},
        ])))
        # Unicode document ids and terms, and a term repeated within a doc.
        Snapshot.load(build_from_lines(lines([
            {"id": "中", "text": "快狐 跳过 快狐 quick"},
        ])))

    def test_document_id_rules(self):
        self.assert_rejected({"version": 1, "documents": [1], "terms": []})
        self.assert_rejected({"version": 1, "documents": [""], "terms": []})
        self.assert_rejected(
            {"version": 1, "documents": ["b", "a"], "terms": []})
        self.assert_rejected(
            {"version": 1, "documents": ["a", "a"], "terms": []})

    def _one_term(self, term, positions, docs=("d",)):
        return {"version": 1, "documents": list(docs), "terms": [
            {"term": term,
             "postings": [{"id": docs[0], "positions": positions}]}]}

    def test_term_rules(self):
        # Non-empty, strictly increasing/unique, analyzer-stable.
        self.assert_rejected(self._one_term("", [0]))
        self.assert_rejected({"version": 1, "documents": ["d"], "terms": [
            {"term": "b", "postings": [{"id": "d", "positions": [0]}]},
            {"term": "a", "postings": [{"id": "d", "positions": [1]}]},
        ]})
        self.assert_rejected({"version": 1, "documents": ["d"], "terms": [
            {"term": "a", "postings": [{"id": "d", "positions": [0]}]},
            {"term": "a", "postings": [{"id": "d", "positions": [1]}]},
        ]})
        # Uppercase casefolds away; punctuation splits; ligature normalizes;
        # whitespace splits into two tokens.
        self.assert_rejected(self._one_term("A", [0]))
        self.assert_rejected(self._one_term("a.b", [0, 1]))
        self.assert_rejected(self._one_term("ﬁle", [0]))
        self.assert_rejected(self._one_term("a b", [0, 1]))

    def test_postings_nonempty_and_doc_references(self):
        self.assert_rejected(
            {"version": 1, "documents": ["d"],
             "terms": [{"term": "a", "postings": []}]})
        # Posting references a document absent from "documents".
        self.assert_rejected({"version": 1, "documents": ["d"], "terms": [
            {"term": "a", "postings": [{"id": "other", "positions": [0]}]}]})
        self.assert_rejected({"version": 1, "documents": ["a", "b"], "terms": [
            {"term": "x", "postings": [
                {"id": "b", "positions": [0]},
                {"id": "a", "positions": [0]},
            ]},
        ]})
        self.assert_rejected({"version": 1, "documents": ["a"], "terms": [
            {"term": "x", "postings": [
                {"id": "a", "positions": [0]},
                {"id": "a", "positions": [1]},
            ]},
        ]})

    def test_position_rules(self):
        self.assert_rejected(self._one_term("a", []))
        self.assert_rejected(self._one_term("a", [True]))
        self.assert_rejected(self._one_term("a", [-1]))
        self.assert_rejected(self._one_term("a", [0.5]))
        self.assert_rejected(self._one_term("a", [1, 0]))
        self.assert_rejected(self._one_term("a", [0, 0]))
        # Positions must be an array of the right element type.
        self.assert_rejected({"version": 1, "documents": ["d"], "terms": [
            {"term": "a", "postings": [{"id": "d", "positions": "0"}]}]})

    def test_cross_term_coverage(self):
        def two(a_pos, b_pos):
            return {"version": 1, "documents": ["d"], "terms": [
                {"term": "a",
                 "postings": [{"id": "d", "positions": a_pos}]},
                {"term": "b",
                 "postings": [{"id": "d", "positions": b_pos}]},
            ]}
        # Same position claimed by two terms.
        self.assert_rejected(two([0], [0]))
        # Gap: union {0, 2} skips position 1.
        self.assert_rejected(two([0], [2]))
        # Coverage not starting at 0.
        self.assert_rejected(self._one_term("a", [1]))

    def test_valid_coverage_still_loads(self):
        # {a:[0,2], b:[1]} is exactly what build emits for "a b a".
        obj = {"version": 1, "documents": ["d"], "terms": [
            {"term": "a", "postings": [{"id": "d", "positions": [0, 2]}]},
            {"term": "b", "postings": [{"id": "d", "positions": [1]}]},
        ]}
        Snapshot.load(json.dumps(obj))

    def test_malformed_shapes(self):
        self.assert_rejected({"version": 1, "documents": [], "terms": [{}]})
        self.assert_rejected({"version": 1, "documents": [], "terms": [
            {"term": "a", "postings": "nope"}]})
        self.assert_rejected({"version": 1, "documents": ["d"], "terms": [
            {"term": "a", "postings": ["nope"]}]})


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.docs_path = os.path.join(self.dir, "docs.jsonl")
        self.snap_path = os.path.join(self.dir, "snap.json")
        with open(self.docs_path, "w", encoding="utf-8") as fp:
            for line in lines(DOCS):
                fp.write(line)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_version_and_help_unchanged(self):
        code, out, _ = self.run_cli("version")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "0.1.0")
        for help_arg in ("help", "-h", "--help"):
            code, out, _ = self.run_cli(help_arg)
            self.assertEqual(code, 0)
            self.assertIn("usage:", out)

    def test_unknown_command(self):
        code, _, err = self.run_cli("frobnicate")
        self.assertEqual(code, 2)
        self.assertIn("unknown command", err)

    def test_build_and_search_roundtrip(self):
        code, _, err = self.run_cli("build", self.docs_path, self.snap_path)
        self.assertEqual(code, 0, err)
        code, out, err = self.run_cli("search", self.snap_path, "quick AND fox")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["a", "b"])
        self.assertTrue(out.endswith("\n"))

    def test_search_empty_query(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        code, out, _ = self.run_cli("search", self.snap_path, "")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "[]")

    def test_output_bytes_stable_and_sorted(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        _, out1, _ = self.run_cli("search", self.snap_path, "quick")
        _, out2, _ = self.run_cli("search", self.snap_path, "quick")
        self.assertEqual(out1, out2)
        ids = json.loads(out1)
        self.assertEqual(ids, sorted(ids))

    def test_build_validation_error_exit_2_and_prefix(self):
        bad = os.path.join(self.dir, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as fp:
            fp.write("{not json\n")
        code, _, err = self.run_cli("build", bad, self.snap_path)
        self.assertEqual(code, 2)
        self.assertTrue(err.splitlines()[0].startswith("error:"))
        self.assertFalse(os.path.exists(self.snap_path))

    def test_search_query_error_exit_2_no_partial_output(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        code, out, err = self.run_cli("search", self.snap_path, "fox AND")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_unsupported_snapshot_version_exit_2(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        with open(self.snap_path, encoding="utf-8") as fp:
            obj = json.load(fp)
        obj["version"] = 42
        other = os.path.join(self.dir, "v42.json")
        with open(other, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
        code, _, err = self.run_cli("search", other, "fox")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_corrupt_snapshot_exit_2_no_partial_output(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        with open(self.snap_path, encoding="utf-8") as fp:
            obj = json.load(fp)
        # Impossible internal structure: an empty positions posting would
        # otherwise still let the term match the document.
        obj["terms"][0]["postings"][0]["positions"] = []
        bad = os.path.join(self.dir, "bad-snap.json")
        with open(bad, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
        code, out, err = self.run_cli("search", bad, "fox")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))
        # A same-position conflict must be rejected the same way.
        with open(self.snap_path, encoding="utf-8") as fp:
            obj = json.load(fp)
        obj["terms"][0]["postings"][0]["positions"] = [0]
        obj["terms"][1]["postings"][0]["positions"] = [0]
        with open(bad, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
        code, out, err = self.run_cli("search", bad, "fox")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_missing_input_exit_1(self):
        code, _, err = self.run_cli(
            "build", os.path.join(self.dir, "nope.jsonl"), self.snap_path)
        self.assertEqual(code, 1)
        self.assertTrue(err.splitlines()[0].startswith("error:"))
        code, _, _ = self.run_cli(
            "search", os.path.join(self.dir, "nope.json"), "fox")
        self.assertEqual(code, 1)

    def test_failed_build_preserves_existing_snapshot(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        with open(self.snap_path, "rb") as fp:
            original = fp.read()
        bad = os.path.join(self.dir, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as fp:
            fp.write('{"id":"z"}\n')  # missing text
        code, _, _ = self.run_cli("build", bad, self.snap_path)
        self.assertEqual(code, 2)
        with open(self.snap_path, "rb") as fp:
            self.assertEqual(fp.read(), original)

    def test_unwritable_output_exit_1(self):
        code, _, err = self.run_cli(
            "build", self.docs_path,
            os.path.join(self.dir, "no_such_dir", "snap.json"))
        self.assertEqual(code, 1)
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_build_rejects_non_utf8(self):
        raw_path = os.path.join(self.dir, "raw.jsonl")
        with open(raw_path, "wb") as fp:
            fp.write(b"\xff\xfe\x00\n")
        code, _, err = self.run_cli("build", raw_path, self.snap_path)
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error:"))

    def test_wrong_arg_count(self):
        self.assertEqual(self.run_cli("build", "only-one")[0], 2)
        self.assertEqual(self.run_cli("search", "only-one")[0], 2)

    def test_docs_sorted_by_codepoint(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        _, out, _ = self.run_cli("search", self.snap_path, "NOT café")
        # Everything but d; "中" comes after ASCII ids.
        self.assertEqual(json.loads(out), ["a", "b", "c", "中"])

    def test_prefix_search_cli(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        # "qu*" hits quick in a, b and 中; "f*" hits fox in a, b plus "file"
        # in d (its "ﬁle" NFKC-normalizes to "file").
        code, out, err = self.run_cli("search", self.snap_path, "qu*")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["a", "b", "中"])
        code, out, _ = self.run_cli(
            "search", self.snap_path, "f* AND NOT qu*")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), ["d"])
        code, out, _ = self.run_cli(
            "search", self.snap_path, "qu* AND fox")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), ["a", "b"])
        # Prefix matching nothing is still a successful empty result.
        code, out, _ = self.run_cli("search", self.snap_path, "zzz*")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "[]")

    def test_prefix_bytes_stable_across_equivalent_snapshots(self):
        # Rebuild the same corpus from a different line order; prefix output
        # must be byte-identical.
        self.run_cli("build", self.docs_path, self.snap_path)
        reversed_path = os.path.join(self.dir, "docs-rev.jsonl")
        with open(reversed_path, "w", encoding="utf-8") as fp:
            for line in lines(list(reversed(DOCS))):
                fp.write(line)
        rev_snap = os.path.join(self.dir, "snap-rev.json")
        code, _, err = self.run_cli("build", reversed_path, rev_snap)
        self.assertEqual(code, 0, err)
        with open(self.snap_path, "rb") as fp:
            self.assertEqual(fp.read(), open(rev_snap, "rb").read())
        for q in ("qu*", "f* OR d*", "NOT c*", "(qu* OR café*) AND fox"):
            _, out1, _ = self.run_cli("search", self.snap_path, q)
            _, out2, _ = self.run_cli("search", rev_snap, q)
            self.assertEqual(out1, out2, q)

    def test_illegal_prefix_exit_2_no_partial_output(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        for q in ("*", "app**", "*fox", "a,b*", "!*", "qu* fox"):
            code, out, err = self.run_cli("search", self.snap_path, q)
            self.assertEqual(code, 2, q)
            self.assertEqual(out, "", q)
            self.assertTrue(err.splitlines()[0].startswith("error:"), q)

    def test_starred_keyword_and_phrase_star_via_cli(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        # AND* is a prefix term ("and..."), matching document b's "and".
        code, out, err = self.run_cli("search", self.snap_path, "AND*")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["b"])
        # The star stays literal inside phrases: "quick*" phrases as "quick".
        code, out, err = self.run_cli("search", self.snap_path, '"quick*"')
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["a", "b", "中"])

    def test_prefix_works_without_rebuilding_old_snapshot(self):
        # Write a v1 snapshot by hand; the new query must read it as-is.
        old = os.path.join(self.dir, "old-v1.json")
        with open(old, "w", encoding="utf-8") as fp:
            json.dump({"version": 1, "documents": ["d1", "d2"], "terms": [
                {"term": "app",
                 "postings": [{"id": "d1", "positions": [0]}]},
                {"term": "apply",
                 "postings": [{"id": "d2", "positions": [0]}]},
            ]}, fp, ensure_ascii=False)
        code, out, err = self.run_cli("search", old, "app*")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["d1", "d2"])


if __name__ == "__main__":
    unittest.main()
