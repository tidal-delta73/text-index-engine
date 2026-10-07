"""End-to-end and unit tests for the build/search vertical slice."""
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from text_index_engine import query
from text_index_engine.analysis import tokenize
from text_index_engine.__main__ import main
from text_index_engine.errors import DataError
from text_index_engine.rank import rank
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


class StreamingBuildTests(unittest.TestCase):
    """build_from_lines on one-shot iterables, byte-compatible with v1."""

    # Canonical bytes captured from the pre-refactor v1 builder for DOCS.
    CANONICAL = (
        b'{"version":1,"documents":["a","b","c","d","\xe4\xb8\xad"],"terms":['
        b'{"term":"a","postings":[{"id":"b","positions":[0,4]}]},'
        b'{"term":"and","postings":[{"id":"b","positions":[3]}]},'
        b'{"term":"brown","postings":[{"id":"a","positions":[2]}]},'
        b'{"term":"caf\xc3\xa9","postings":[{"id":"d","positions":[0,1]}]},'
        b'{"term":"dog","postings":[{"id":"a","positions":[8]},'
        b'{"id":"b","positions":[6]}]},'
        b'{"term":"file","postings":[{"id":"d","positions":[2]}]},'
        b'{"term":"fox","postings":[{"id":"a","positions":[3]},'
        b'{"id":"b","positions":[2]}]},'
        b'{"term":"jumps","postings":[{"id":"a","positions":[4]}]},'
        b'{"term":"lazy","postings":[{"id":"a","positions":[7]}]},'
        b'{"term":"over","postings":[{"id":"a","positions":[5]}]},'
        b'{"term":"quick","postings":[{"id":"a","positions":[1]},'
        b'{"id":"b","positions":[1,5]},'
        b'{"id":"\xe4\xb8\xad","positions":[3]}]},'
        b'{"term":"the","postings":[{"id":"a","positions":[0,6]}]},'
        b'{"term":"\xe5\xbf\xab\xe7\x8b\x90","postings":['
        b'{"id":"\xe4\xb8\xad","positions":[0]}]},'
        b'{"term":"\xe6\x87\x92\xe7\x8b\x97","postings":['
        b'{"id":"\xe4\xb8\xad","positions":[2]}]},'
        b'{"term":"\xe8\xb7\xb3\xe8\xbf\x87","postings":['
        b'{"id":"\xe4\xb8\xad","positions":[1]}]}]}\n'
    )

    def test_canonical_bytes_baseline(self):
        self.assertEqual(build_from_lines(lines(DOCS)), self.CANONICAL)

    def test_one_shot_iterables(self):
        # No len, no indexing, no re-iteration: a plain generator and an
        # iterator must both yield the canonical bytes.
        gen = (line for line in lines(DOCS))
        self.assertEqual(build_from_lines(gen), self.CANONICAL)
        self.assertEqual(build_from_lines(iter(lines(DOCS))), self.CANONICAL)

    def test_input_consumed_lazily_in_one_pass(self):
        # The builder must not ask for len()/getitem or restart the input.
        class OneShot:
            def __init__(self, items):
                self._it = iter(items)
                self.pulls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.pulls += 1
                return next(self._it)

        src = OneShot(lines(DOCS))
        self.assertEqual(build_from_lines(src), self.CANONICAL)
        self.assertEqual(src.pulls, len(DOCS) + 1)  # exactly one pass

    def test_shuffled_corpus_with_empty_docs(self):
        # Empty-text doc "c" shuffled to the front; bytes stay canonical.
        shuffled = [DOCS[2], DOCS[4], DOCS[1], DOCS[0], DOCS[3]]
        self.assertEqual(build_from_lines(iter(lines(shuffled))),
                         self.CANONICAL)
        # A corpus of only empty/separator-only documents.
        empties = [
            {"id": "x", "text": ""},
            {"id": "y", "text": "  - ! "},
        ]
        obj = json.loads(build_from_lines(iter(lines(empties))))
        self.assertEqual(obj["documents"], ["x", "y"])
        self.assertEqual(obj["terms"], [])

    def test_late_failures_raise_without_partial_output(self):
        good = lines(DOCS)
        # Duplicate id on the last line of a one-shot generator.
        dup = good + [json.dumps({"id": "a", "text": "again"}) + "\n"]
        with self.assertRaises(DataError) as ctx:
            build_from_lines(iter(dup))
        self.assertIn(f"line {len(dup)}", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception))
        # Malformed JSON on the last line.
        bad_tail = good + ["{not json\n"]
        with self.assertRaises(DataError) as ctx:
            build_from_lines(iter(bad_tail))
        self.assertIn(f"line {len(bad_tail)}", str(ctx.exception))
        # Iterator that fails mid-stream: the exception propagates and no
        # bytes are returned.
        def dying():
            yield good[0]
            raise DataError("line 2: boom")
        with self.assertRaises(DataError):
            build_from_lines(dying())


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


class PrefixQueryTests(unittest.TestCase):
    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(lines(DOCS)))

    def run_query(self, q):
        node = query.parse(q)
        return sorted(query.evaluate(node, self.snap))

    def test_prefix_ast_shape(self):
        self.assertEqual(query.parse("app*"), ("prefix", "app"))

    def test_prefix_matches_every_term_starting_with_it(self):
        # "quick" appears in a, b and 中.
        self.assertEqual(self.run_query("qu*"), ["a", "b", "中"])
        # fox -> a,b and file -> d both start with "f".
        self.assertEqual(self.run_query("f*"), ["a", "b", "d"])
        # An exact-word prefix behaves like the plain term.
        self.assertEqual(self.run_query("dog*"), ["a", "b"])
        # Single-character prefix.
        self.assertEqual(self.run_query("l*"), ["a"])
        # Prefix of a term that exists only alongside longer words.
        self.assertEqual(self.run_query("caf"), [])      # no wildcard: term
        self.assertEqual(self.run_query("caf*"), ["d"])  # café

    def test_prefix_is_normalized_like_terms(self):
        self.assertEqual(self.run_query("QU*"), ["a", "b", "中"])  # casefold
        self.assertEqual(self.run_query("CAF*"), ["d"])            # casefold
        self.assertEqual(self.run_query("ﬁ*"), ["d"])              # fi -> file
        # Fullwidth ASCII NFKC-folds to plain letters before prefixing.
        self.assertEqual(
            sorted(query.evaluate(
                query.parse("ＱＵ*"), self.snap)),
            ["a", "b", "中"],
        )

    def test_cjk_prefix_is_codepoint_prefix(self):
        self.assertEqual(self.run_query("快*"), ["中"])   # 快狐
        self.assertEqual(self.run_query("懒*"), ["中"])   # 懒狗
        self.assertEqual(self.run_query("狐*"), [])       # second codepoint only

    def test_prefix_with_no_matching_term_is_empty_set(self):
        self.assertEqual(self.run_query("zzz*"), [])
        # NFKC of ① is "1", which prefixes no indexed term either.
        self.assertEqual(self.run_query("①*"), [])

    def test_prefix_in_boolean_combinations(self):
        self.assertEqual(self.run_query("qu* AND fox"), ["a", "b"])
        self.assertEqual(self.run_query("qu* OR café"), ["a", "b", "d", "中"])
        self.assertEqual(self.run_query("NOT qu*"), ["c", "d"])
        self.assertEqual(
            self.run_query("(qu* OR laz*) AND dog"), ["a", "b"])
        self.assertEqual(self.run_query("zzz* OR fox"), ["a", "b"])
        self.assertEqual(self.run_query("fox AND zzz*"), [])
        # NOT over a matching/non-matching prefix; empty-text doc c included.
        self.assertEqual(
            self.run_query("NOT f*"), ["c", "中"])
        self.assertEqual(
            self.run_query("NOT zzz*"), ["a", "b", "c", "d", "中"])
        self.assertEqual(self.run_query("NOT NOT*"),
                         ["a", "b", "c", "d", "中"])
        self.assertEqual(self.run_query("(qu*)"), ["a", "b", "中"])

    def test_starred_keywords_are_prefix_terms(self):
        # "and" is an ordinary word in document b.
        self.assertEqual(self.run_query("AND*"), ["b"])
        self.assertEqual(self.run_query("and*"), ["b"])
        # No term begins with "or"/"not" in this corpus.
        self.assertEqual(self.run_query("OR*"), [])
        self.assertEqual(self.run_query("NOT*"), [])
        # Bare uppercase keywords keep their operator meaning.
        with self.assertRaises(DataError):
            query.parse("AND fox")
        # A starred keyword still composes as an operand.
        self.assertEqual(self.run_query("NOT AND*"),
                         ["a", "c", "d", "中"])

    def test_star_inside_phrase_has_no_wildcard_meaning(self):
        # Tokenized by ordinary phrase rules: the star is punctuation, and
        # the resulting one-word phrase matches like the bare phrase.
        self.assertEqual(self.run_query('"quick*"'), ["a", "b", "中"])
        # "dog*fox" phrase-analyzes to two words that are not consecutive.
        self.assertEqual(self.run_query('"dog*fox"'), [])
        # A lone star inside quotes is still a phrase with no searchable term.
        with self.assertRaises(DataError):
            query.parse('"*"')

    def test_prefix_adjacency_without_operator_rejected(self):
        for q in ("fox qu*", "qu* fox", '"quick fox" qu*', 'qu* "fox"',
                  "qu* dog*"):
            with self.assertRaises(DataError):
                query.parse(q)

    def assert_rejected(self, q):
        with self.assertRaises(DataError) as ctx:
            node = query.parse(q)
            query.evaluate(node, self.snap)
        self.assertTrue(str(ctx.exception).startswith("query:"))

    def test_illegal_star_forms_rejected(self):
        # Lone star.
        self.assert_rejected("*")
        # More than one star / star not trailing the bare term.
        self.assert_rejected("**")
        self.assert_rejected("foo**")
        self.assert_rejected("*foo")
        self.assert_rejected("foo*bar")
        self.assert_rejected("fo*o*")
        self.assert_rejected("a*b*")
        # Star separated by whitespace is a lone-star run.
        self.assert_rejected("foo *")
        self.assert_rejected("* foo")
        # Prefix body analyzes to zero terms.
        self.assert_rejected("!!!*")
        self.assert_rejected("-*")
        # Prefix body analyzes to several terms inside one bare run.
        self.assert_rejected("foo,bar*")
        self.assert_rejected("quick-fox*")
        # A star dangling after a closing paren is its own bad run.
        self.assert_rejected("(fox)*")


class PrefixSnapshotTests(unittest.TestCase):
    def test_prefix_results_independent_of_document_input_order(self):
        # Build equivalent snapshots from different input orders; prefix
        # queries must return identical, sorted hit sets against each one.
        orders = [
            DOCS,
            list(reversed(DOCS)),
            DOCS[2:] + DOCS[:2],
        ]
        snaps = [Snapshot.load(build_from_lines(lines(order)))
                 for order in orders]
        # The snapshots themselves are byte-identical already, but the
        # guarantee asked for is identical *query results*:
        self.assertTrue(all(
            build_from_lines(lines(order)) == build_from_lines(lines(DOCS))
            for order in orders))
        prefix_queries = [
            "qu*", "f*", "caf*", "快*", "zzz*",
            "NOT f*", "qu* AND fox", "(qu* OR laz*) AND dog",
        ]
        for q in prefix_queries:
            results = [
                sorted(query.evaluate(query.parse(q), snap))
                for snap in snaps
            ]
            self.assertEqual(results[0], results[1], q)
            self.assertEqual(results[0], results[2], q)

    def test_old_v1_snapshot_needs_no_rebuild(self):
        # A hand-written v1 snapshot (same format build has always emitted)
        # must serve prefix queries without any migration or rebuild.
        obj = {
            "version": 1,
            "documents": ["d1", "d2"],
            "terms": [
                {"term": "app",
                 "postings": [{"id": "d1", "positions": [0]}]},
                {"term": "apple",
                 "postings": [{"id": "d2", "positions": [0]}]},
                {"term": "banana",
                 "postings": [{"id": "d1", "positions": [1]}]},
            ],
        }
        snap = Snapshot.load(json.dumps(obj, ensure_ascii=False))
        self.assertEqual(
            sorted(query.evaluate(query.parse("app*"), snap)),
            ["d1", "d2"],
        )
        self.assertEqual(
            sorted(query.evaluate(query.parse("APP*"), snap)),
            ["d1", "d2"],
        )
        self.assertEqual(
            sorted(query.evaluate(query.parse("b*"), snap)), ["d1"])
        self.assertEqual(
            sorted(query.evaluate(query.parse("appz*"), snap)), [])
        # Plain term behavior on the same snapshot is untouched.
        self.assertEqual(
            sorted(query.evaluate(query.parse("app"), snap)), ["d1"])


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
        code, out, err = self.run_cli("search", self.snap_path, "qu*")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), ["a", "b", "中"])
        self.assertTrue(out.endswith("\n"))

    def test_prefix_search_byte_stable(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        _, out1, _ = self.run_cli("search", self.snap_path, "f*")
        _, out2, _ = self.run_cli("search", self.snap_path, "f*")
        self.assertEqual(out1, out2)
        self.assertEqual(out1, '["a","b","d"]\n')

    def test_prefix_no_match_prints_empty_array(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        code, out, err = self.run_cli("search", self.snap_path, "nope*")
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "[]\n")

    def test_prefix_query_error_exit_2_no_partial_output(self):
        self.run_cli("build", self.docs_path, self.snap_path)
        for bad in ("*", "foo**", "*foo", "foo*bar", "!!!*", "foo,bar*",
                    "qu* fox"):
            code, out, err = self.run_cli("search", self.snap_path, bad)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", bad)
            self.assertTrue(err.splitlines()[0].startswith("error:"), bad)


def bm25_expected(snap_obj, terms, candidates, k1=1.2, b=0.75):
    """Independent BM25 expectation derived from the v1 snapshot object.

    Reads only the documented snapshot format (documents/terms/postings) and
    applies the public formula; never calls text_index_engine.rank.
    ``terms`` is the hand-listed multiset of contributing term occurrences.
    """
    doc_ids = snap_obj["documents"]
    n = len(doc_ids)
    postings = {t["term"]: {p["id"]: p["positions"] for p in t["postings"]}
                for t in snap_obj["terms"]}
    dl = {d: 0 for d in doc_ids}
    for posting in postings.values():
        for d, positions in posting.items():
            dl[d] += len(positions)
    avgdl = sum(dl.values()) / n if n else 0
    rows = []
    for d in candidates:
        total = 0.0
        if avgdl > 0:
            for term in terms:
                df = len(postings.get(term, {}))
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = len(postings.get(term, {}).get(d, []))
                total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[d]
                                                          / avgdl))
        rows.append((d, f"{total:.6f}"))
    rows.sort(key=lambda row: (-float(row[1]), row[0]))
    return [{"id": d, "score": s} for d, s in rows]


class RankTests(unittest.TestCase):
    def setUp(self):
        self.snap_obj = json.loads(build_from_lines(lines(DOCS)))
        self.snap = Snapshot.load(json.dumps(self.snap_obj, ensure_ascii=False))

    def run_rank(self, q):
        node = query.parse(q)
        return rank(node, self.snap) if node is not None else []

    def assert_rank(self, q, terms, candidates):
        expected = bm25_expected(self.snap_obj, terms, candidates)
        self.assertEqual(self.run_rank(q), expected)
        return expected

    def test_single_term_scores_match_formula(self):
        expected = self.assert_rank("quick", ["quick"], ["a", "b", "中"])
        # Scores are genuine BM25 values: all three differ here (dl and tf
        # vary), are positive and carry exactly six decimals.
        scores = [row["score"] for row in expected]
        self.assertEqual(len(set(scores)), 3)
        for row in expected:
            self.assertRegex(row["score"], r"^\d+\.\d{6}$")
            self.assertGreater(float(row["score"]), 0.0)
        # Higher tf wins: b has quick twice, a and 中 once; a is longer
        # than 中, so 中 outranks a.
        self.assertEqual([row["id"] for row in expected], ["b", "中", "a"])

    def test_empty_query_and_empty_candidates(self):
        self.assertEqual(self.run_rank(""), [])
        self.assertEqual(self.run_rank("   "), [])
        self.assertEqual(self.run_rank("zzz"), [])
        self.assertEqual(self.run_rank("zzz*"), [])

    def test_not_only_query_scores_zero_in_id_order(self):
        # No positive leaf: every candidate scores 0.000000, id ordered.
        self.assertEqual(
            self.run_rank("NOT fox"),
            [{"id": "c", "score": "0.000000"},
             {"id": "d", "score": "0.000000"},
             {"id": "中", "score": "0.000000"}],
        )

    def test_even_not_depth_scores_like_positive(self):
        self.assertEqual(self.run_rank("NOT NOT quick"), self.run_rank("quick"))
        self.assertEqual(
            self.run_rank("NOT NOT NOT NOT fox"), self.run_rank("fox"))

    def test_odd_not_leaf_filters_but_never_scores(self):
        # Candidates: quick docs minus fox docs = {中}; only quick scores.
        self.assert_rank("quick AND NOT fox", ["quick"], ["中"])
        # NOT over a group: every leaf inside is at odd depth.
        self.assert_rank("quick AND NOT (fox OR café)", ["quick"], ["中"])

    def test_phrase_contributes_each_word_without_bonus(self):
        self.assert_rank('"quick fox"', ["quick", "fox"], ["b"])
        # A one-word phrase scores exactly like the bare term.
        self.assertEqual(self.run_rank('"quick"'), self.run_rank("quick"))

    def test_prefix_contributes_each_expanded_term(self):
        # qu* expands to exactly "quick".
        self.assert_rank("qu*", ["quick"], ["a", "b", "中"])
        # f* expands to file and fox; the union matches a, b and d.
        self.assert_rank("f*", ["file", "fox"], ["a", "b", "d"])
        # A prefix matching no dictionary term has no scoring terms at all.
        self.assertEqual(
            self.run_rank("NOT zzz*"),
            [{"id": d, "score": "0.000000"} for d in ["a", "b", "c", "d", "中"]],
        )

    def test_repeated_term_occurrences_accumulate(self):
        single = self.run_rank("quick")
        double = self.run_rank("quick OR quick")
        self.assertEqual([r["id"] for r in single], [r["id"] for r in double])
        # Exact strings checked against the independent formula: each
        # occurrence of the same term scores again.
        self.assert_rank("quick OR quick", ["quick", "quick"], ["a", "b", "中"])

    def test_ties_break_by_document_id_codepoint_order(self):
        docs = [{"id": "b", "text": "fox"}, {"id": "a", "text": "fox"},
                {"id": "中", "text": "fox"}]
        snap = Snapshot.load(build_from_lines(lines(docs)))
        rows = rank(query.parse("fox"), snap)
        self.assertEqual([r["id"] for r in rows], ["a", "b", "中"])
        self.assertTrue(all(r["score"] == rows[0]["score"] for r in rows))

    def test_avgdl_zero_scores_all_candidates_zero(self):
        docs = [{"id": "x", "text": ""}, {"id": "y", "text": "  - ! "}]
        snap = Snapshot.load(build_from_lines(lines(docs)))
        self.assertEqual(
            rank(query.parse("NOT fox"), snap),
            [{"id": "x", "score": "0.000000"},
             {"id": "y", "score": "0.000000"}],
        )
        # A term query has no candidates at all.
        self.assertEqual(rank(query.parse("fox"), snap), [])

    def test_n_zero_returns_empty(self):
        snap = Snapshot.load(build_from_lines([]))
        self.assertEqual(rank(query.parse("fox"), snap), [])
        self.assertEqual(rank(query.parse("NOT fox"), snap), [])

    def test_old_v1_snapshot_ranks_without_rebuild(self):
        obj = {
            "version": 1,
            "documents": ["d1", "d2"],
            "terms": [
                {"term": "app",
                 "postings": [{"id": "d1", "positions": [0]}]},
                {"term": "apple",
                 "postings": [{"id": "d2", "positions": [0]}]},
                {"term": "banana",
                 "postings": [{"id": "d1", "positions": [1]}]},
            ],
        }
        snap = Snapshot.load(json.dumps(obj, ensure_ascii=False))
        rows = rank(query.parse("app*"), snap)
        # app* expands to app and apple; d1 scores via app, d2 via apple.
        self.assertEqual(rows, bm25_expected(obj, ["app", "apple"],
                                             ["d1", "d2"]))
        self.assertEqual([r["id"] for r in rows], ["d2", "d1"])  # dl 1 < 2
        self.assertEqual(
            rank(query.parse("banana"), snap),
            bm25_expected(obj, ["banana"], ["d1"]))

    def test_deep_not_chain_parity_at_low_recursion_limit(self):
        saved = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            even = rank(query.parse("NOT " * 10000 + "quick"), self.snap)
            odd = rank(query.parse("NOT " * 10001 + "quick"), self.snap)
        finally:
            sys.setrecursionlimit(saved)
        self.assertEqual(even, self.run_rank("quick"))
        self.assertEqual(
            odd, [{"id": d, "score": "0.000000"} for d in ["c", "d"]])


class RankCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.docs_path = os.path.join(self.dir, "docs.jsonl")
        self.snap_path = os.path.join(self.dir, "snap.json")
        with open(self.docs_path, "w", encoding="utf-8") as fp:
            for line in lines(DOCS):
                fp.write(line)
        code, _, err = self.run_cli(
            "build", self.docs_path, self.snap_path)
        self.assertEqual(code, 0, err)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_help_lists_rank(self):
        for help_arg in ("help", "-h", "--help"):
            code, out, _ = self.run_cli(help_arg)
            self.assertEqual(code, 0)
            self.assertIn("rank <snapshot>", out)

    def test_rank_output_shape_and_trailing_newline(self):
        code, out, err = self.run_cli("rank", self.snap_path, "quick")
        self.assertEqual(code, 0, err)
        self.assertTrue(out.endswith("\n") and out.count("\n") == 1)
        rows = json.loads(out)
        self.assertEqual([r["id"] for r in rows], ["b", "中", "a"])
        for row in rows:
            self.assertEqual(sorted(row), ["id", "score"])
            self.assertRegex(row["score"], r"^\d+\.\d{6}$")
        # Compact separators, id before score, non-ASCII kept literal.
        self.assertTrue(out.startswith('[{"id":"b","score":"'))
        self.assertIn('"id":"中"', out)

    def test_rank_byte_stable_and_matches_module(self):
        _, out1, _ = self.run_cli("rank", self.snap_path, "f* OR café")
        _, out2, _ = self.run_cli("rank", self.snap_path, "f* OR café")
        self.assertEqual(out1, out2)
        snap = Snapshot.load(build_from_lines(lines(DOCS)))
        module_rows = rank(query.parse("f* OR café"), snap)
        self.assertEqual(json.loads(out1), module_rows)

    def test_rank_empty_query_and_no_hits_print_empty_array(self):
        for q in ("", "zzz"):
            code, out, err = self.run_cli("rank", self.snap_path, q)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "[]\n")

    def test_rank_not_only_query_zero_scores(self):
        code, out, err = self.run_cli("rank", self.snap_path, "NOT fox")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            json.loads(out),
            [{"id": "c", "score": "0.000000"},
             {"id": "d", "score": "0.000000"},
             {"id": "中", "score": "0.000000"}],
        )

    def test_rank_wrong_arg_count_exit_2(self):
        for argv in (("rank",), ("rank", "only-one"),
                     ("rank", "a", "b", "c")):
            code, out, err = self.run_cli(*argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "", argv)
            self.assertTrue(err.splitlines()[0].startswith("error:"), argv)

    def test_rank_query_error_exit_2_no_partial_output(self):
        for bad in ("fox AND", "*", '"unterminated', "()", "fox dog"):
            code, out, err = self.run_cli("rank", self.snap_path, bad)
            self.assertEqual(code, 2, bad)
            self.assertEqual(out, "", bad)
            self.assertTrue(err.splitlines()[0].startswith("error:"), bad)

    def test_rank_corrupt_snapshot_exit_2(self):
        with open(self.snap_path, encoding="utf-8") as fp:
            obj = json.load(fp)
        obj["version"] = 42
        bad = os.path.join(self.dir, "bad.json")
        with open(bad, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
        code, out, err = self.run_cli("rank", bad, "fox")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))

    def test_rank_missing_snapshot_exit_1(self):
        code, out, err = self.run_cli(
            "rank", os.path.join(self.dir, "nope.json"), "fox")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.splitlines()[0].startswith("error:"))


class DeepQueryTests(unittest.TestCase):
    """Depth is bounded by the query language, not the recursion limit."""

    N = 10000
    LOW_LIMIT = 100

    def setUp(self):
        self.snap = Snapshot.load(build_from_lines(lines(DOCS)))

    def _eval(self, q):
        node = query.parse(q)
        return sorted(query.evaluate(node, self.snap)) if node is not None else []

    def _with_low_limit(self, fn):
        saved = sys.getrecursionlimit()
        sys.setrecursionlimit(self.LOW_LIMIT)
        try:
            return fn()
        finally:
            sys.setrecursionlimit(saved)

    def test_deep_not_chain(self):
        fox = ["a", "b"]
        not_fox = ["c", "d", "中"]
        # An even chain is the identity; an odd one is the complement.
        self.assertEqual(
            self._with_low_limit(lambda: self._eval("NOT " * self.N + "fox")),
            fox)
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("NOT " * (self.N + 1) + "fox")),
            not_fox)
        # Repeated evaluation is stable.
        q = "NOT " * (self.N + 1) + "fox"
        self.assertEqual(
            self._with_low_limit(lambda: self._eval(q)),
            self._with_low_limit(lambda: self._eval(q)))

    def test_deep_parentheses(self):
        n = self.N
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("(" * n + "fox" + ")" * n)),
            ["a", "b"])
        # Parentheses still override precedence when buried deeply.
        self.assertEqual(
            self._with_low_limit(lambda: self._eval(
                "fox OR " + "(" * n + "(quick AND dog)" + ")" * n)),
            self._eval("fox OR (quick AND dog)"))
        self.assertEqual(
            self._with_low_limit(lambda: self._eval(
                "(" * n + "(fox OR quick) AND dog" + ")" * n)),
            self._eval("(fox OR quick) AND dog"))

    def test_deep_binary_chains(self):
        n = self.N
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("fox" + " AND fox" * (n - 1))),
            ["a", "b"])
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("zzz" + " OR fox" * (n - 1))),
            ["a", "b"])
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("fox" + " AND zzz" * (n - 1))),
            [])
        # Equivalent short and long spellings agree.
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("fox" + " OR fox" * (n - 1))),
            self._eval("fox OR fox"))

    def test_deep_mixed_structure(self):
        n = self.N // 2
        self.assertEqual(
            self._with_low_limit(lambda: self._eval(
                "(" * n + 'NOT "quick fox" OR qu*' + ")" * n)),
            self._eval('(NOT "quick fox" OR qu*)'))
        # NOT over an absent term deep inside still yields the universe.
        universe = ["a", "b", "c", "d", "中"]
        self.assertEqual(
            self._with_low_limit(
                lambda: self._eval("NOT " * (self.N + 1) + "zzz")),
            universe)
        self.assertIn(
            "c",
            set(self._with_low_limit(lambda: self._eval(
                "(" * self.N + "NOT fox" + ")" * self.N))))

    def test_deep_malformed_raises_exactly_data_error(self):
        n = self.N
        bad = [
            "(" * n + "fox",                              # unclosed group
            "(" * n + "fox" + ")" * (n - 1),              # one close short
            "(" * n + "fox AND" + ")" * n,                # trailing operator
            "NOT " * n,                                   # missing operand
            "(" * n + "fox dog" + ")" * n,                # adjacent operands
            "(" * n + 'fox AND "unterminated',            # open phrase at depth
            "(" * n + "NOT" + ")" * n,                    # NOT then close
            "(" * n + "()" + ")" * (n - 1),               # empty group deep
            "(" * n + "AND fox" + ")" * n,                # binary first
            "fox" + " AND fox" * (n - 1) + " AND",        # trailing chain op
        ]

        def run_all():
            for q in bad:
                try:
                    node = query.parse(q)
                    if node is not None:
                        query.evaluate(node, self.snap)
                except DataError as exc:
                    self.assertIs(type(exc), DataError)
                    continue
                self.fail(f"deep malformed query was accepted: {q[:20]!r}")

        self._with_low_limit(run_all)

    def test_deep_malformed_cli_exit_2(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        snap_path = os.path.join(tmp.name, "snap.json")
        with open(os.path.join(tmp.name, "docs.jsonl"), "w",
                  encoding="utf-8") as fp:
            for line in lines(DOCS):
                fp.write(line)
        code = main(["build",
                     os.path.join(tmp.name, "docs.jsonl"),
                     snap_path])
        self.assertEqual(code, 0)
        n = self.N
        for bad in ("(" * n + "fox",
                    "NOT " * n,
                    "(" * n + "fox AND" + ")" * n,
                    "(" * n + '"unterminated'):
            out, err_buf = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err_buf):
                code = main(["search", snap_path, bad])
            self.assertEqual(code, 2, bad[:20])
            self.assertEqual(out.getvalue(), "", bad[:20])
            self.assertTrue(
                err_buf.getvalue().splitlines()[0].startswith("error:"),
                bad[:20])


if __name__ == "__main__":
    unittest.main()
