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


class SnapshotLoadTests(unittest.TestCase):
    def test_unsupported_version(self):
        raw = bytearray(build_from_lines(lines(DOCS)))
        obj = json.loads(raw)
        obj["version"] = 999
        with self.assertRaises(DataError):
            Snapshot.load(json.dumps(obj))

    def test_garbage(self):
        with self.assertRaises(DataError):
            Snapshot.load(b"not json")
        with self.assertRaises(DataError):
            Snapshot.load(b"\xff\xfe")

    def base_obj(self, docs=None):
        return json.loads(build_from_lines(lines(
            DOCS if docs is None else docs)))

    def assert_rejected(self, obj):
        with self.assertRaises(DataError):
            Snapshot.load(json.dumps(obj, ensure_ascii=False))

    def test_built_snapshots_accepted(self):
        # Normal set, empty set, only empty-text docs, unicode ids/terms.
        Snapshot.load(build_from_lines(lines(DOCS)))
        Snapshot.load(build_from_lines([]))
        empty_only = [
            {"id": "a", "text": ""},
            {"id": "中", "text": "   !  "},
        ]
        Snapshot.load(build_from_lines(lines(empty_only)))
        unicode_docs = [{"id": "中", "text": "快狐 跳过"}]
        Snapshot.load(build_from_lines(lines(unicode_docs)))

    def test_document_ids(self):
        obj = self.base_obj()
        obj["documents"].append("")
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["documents"].append("a")
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["documents"] = ["b", "a"]
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["documents"] = ["a", 1]
        self.assert_rejected(obj)

    def test_terms_sorted_and_self_normalizing(self):
        obj = self.base_obj()
        obj["terms"] = list(reversed(obj["terms"]))
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["terms"][0]["term"] = "FOX"  # tokenize("FOX") == ["fox"]
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["terms"][0]["term"] = ""
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["terms"][0]["term"] = "quick fox"  # splits into two tokens
        self.assert_rejected(obj)

        obj = self.base_obj()
        obj["terms"] = [dict(obj["terms"][0]), dict(obj["terms"][0])]
        self.assert_rejected(obj)

    def test_postings(self):
        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"] = []
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][0]["positions"] = []
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"] = list(reversed(term["postings"]))
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"].append(dict(term["postings"][0]))
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][0]["id"] = "ghost"
        self.assert_rejected(obj)

    def test_positions(self):
        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][1]["positions"] = [5, 1]  # out of order
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][1]["positions"] = [1, 1]  # duplicate
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][0]["positions"] = [True]
        self.assert_rejected(obj)

        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"][0]["positions"] = [-1]
        self.assert_rejected(obj)

    def test_per_document_consistency(self):
        # Position hole: drop the term covering position 0 of a 3-token doc.
        hole = json.loads(build_from_lines(
            lines([{"id": "d", "text": "x y z"}])))
        hole["terms"] = [t for t in hole["terms"] if t["term"] != "x"]
        self.assert_rejected(hole)

        # Union must start at 0: only position 1 present is not buildable.
        gap = json.loads(build_from_lines(
            lines([{"id": "d", "text": "x y"}])))
        xt = next(t for t in gap["terms"] if t["term"] == "x")
        xt["postings"][0]["positions"] = [2]
        self.assert_rejected(gap)

        # Same position claimed by two terms is not buildable.
        coll = json.loads(build_from_lines(
            lines([{"id": "d", "text": "one two"}])))
        two = next(t for t in coll["terms"] if t["term"] == "two")
        two["postings"][0]["positions"] = [0]  # collides with "one"
        self.assert_rejected(coll)

        # Snapshot records no original text, so the loader identifies empty
        # documents structurally: a posting covering 0..max makes the doc a
        # legitimate one-token document (buildable from other input text)...
        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "quick")
        term["postings"].append({"id": "c", "positions": [0]})
        term["postings"].sort(key=lambda p: p["id"])
        Snapshot.load(json.dumps(obj, ensure_ascii=False))
        # ...but a non-contiguous fabricated posting is still rejected.
        term["postings"][-1]["positions"] = [1]
        self.assert_rejected(obj)

    def test_empty_positions_does_not_match(self):
        # A posting whose positions list is empty must be rejected outright
        # rather than silently matching the document.
        obj = self.base_obj()
        term = next(t for t in obj["terms"] if t["term"] == "fox")
        term["postings"][0]["positions"] = []
        self.assert_rejected(obj)


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


if __name__ == "__main__":
    unittest.main()
