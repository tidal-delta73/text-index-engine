"""Deterministic inverted-index snapshot: build, serialize, load.

Canonical JSON representation (versioned)::

    {"version":1,"documents":["id",...],"terms":[
      {"term":"...","postings":[{"id":"...","positions":[0,2]}]}]}

Ordering is by Unicode code point throughout: documents and postings by the
original document id, terms by the normalized term. Serialization never
embeds timestamps, paths or other environment data, so the same document set
produces byte-identical snapshots regardless of input line order.
"""
import json
import os
import tempfile

from .analysis import tokenize
from .errors import DataError

SNAPSHOT_VERSION = 1

# Compact canonical form. Key order is fixed by construction; ensure_ascii
# keeps non-ASCII codepoints literal (stable across environments).
_SEPARATORS = (",", ":")


def _encode(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=_SEPARATORS)
            + "\n").encode("utf-8")


def build_from_lines(lines) -> bytes:
    """Validate JSONL document lines and return canonical snapshot bytes.

    Each line must be a JSON object with non-empty unique string ``id`` and
    string ``text``. Raises DataError on any validation failure; nothing is
    written until the whole input has been validated.
    """
    # id -> tokens (dict also doubles as insertion-ordered duplicate check).
    documents: dict[str, list[str]] = {}
    seen_ids: set[str] = set()

    for lineno, line in enumerate(lines, start=1):
        try:
            doc = json.loads(line)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DataError(f"line {lineno}: not a JSON document: {exc.msg}")
        if not isinstance(doc, dict):
            raise DataError(f"line {lineno}: document must be a JSON object")
        if "id" not in doc:
            raise DataError(f"line {lineno}: missing field 'id'")
        if "text" not in doc:
            raise DataError(f"line {lineno}: missing field 'text'")
        doc_id = doc["id"]
        text = doc["text"]
        if not isinstance(doc_id, str):
            raise DataError(f"line {lineno}: 'id' must be a string")
        if not isinstance(text, str):
            raise DataError(f"line {lineno}: 'text' must be a string")
        if doc_id == "":
            raise DataError(f"line {lineno}: 'id' must be non-empty")
        if doc_id in seen_ids:
            raise DataError(f"line {lineno}: duplicate id {doc_id!r}")
        seen_ids.add(doc_id)
        documents[doc_id] = tokenize(text)

    return _encode(_build_object(documents))


def _build_object(documents: dict[str, list[str]]):
    # term -> doc id -> positions
    terms: dict[str, dict[str, list[int]]] = {}
    for doc_id, tokens in documents.items():
        for position, term in enumerate(tokens):
            terms.setdefault(term, {}).setdefault(doc_id, []).append(position)

    term_objs = []
    for term in sorted(terms):
        postings = terms[term]
        term_objs.append({
            "term": term,
            "postings": [
                {"id": doc_id, "positions": postings[doc_id]}
                for doc_id in sorted(postings)
            ],
        })

    return {
        "version": SNAPSHOT_VERSION,
        "documents": sorted(documents),
        "terms": term_objs,
    }


def write_atomic(path: str, data: bytes) -> None:
    """Write bytes to path, replacing the target only after success.

    The temp file is created in the destination directory and fsynced before
    the atomic replace; on any failure the temp file is removed and any
    pre-existing target is left untouched.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class Snapshot:
    """Loaded snapshot used by the search evaluator."""

    def __init__(self, doc_ids: list[str], postings: dict[str, dict[str, list[int]]]):
        self.doc_ids = doc_ids                      # already codepoint-sorted
        self.all_docs = frozenset(doc_ids)
        self.postings = postings                    # term -> doc id -> positions

    @classmethod
    def load(cls, raw: bytes | str) -> "Snapshot":
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"snapshot is not valid JSON: {exc}")
        if not isinstance(obj, dict):
            raise DataError("snapshot: top-level value must be an object")
        version = obj.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise DataError("snapshot: missing or invalid 'version'")
        if version != SNAPSHOT_VERSION:
            raise DataError(
                f"unsupported snapshot version: {version} "
                f"(supported: {SNAPSHOT_VERSION})")

        doc_ids = obj.get("documents")
        terms = obj.get("terms")
        if not isinstance(doc_ids, list) or not isinstance(terms, list):
            raise DataError("snapshot: 'documents' and 'terms' must be arrays")

        # Documents: non-empty strings, unique, strictly increasing by
        # Unicode code point (the order build emits).
        seen: set[str] = set()
        prev_id = None
        for doc_id in doc_ids:
            if not isinstance(doc_id, str):
                raise DataError("snapshot: document ids must be strings")
            if doc_id == "":
                raise DataError("snapshot: document ids must be non-empty")
            if prev_id is not None and not doc_id > prev_id:
                raise DataError(
                    f"snapshot: document ids must be unique and strictly "
                    f"increasing by code point ({doc_id!r} after {prev_id!r})")
            seen.add(doc_id)
            prev_id = doc_id

        postings: dict[str, dict[str, list[int]]] = {}
        # Per document: every indexed position maps to the single term that
        # occupies it. Used to reject same-position multi-term postings and
        # non-continuous position coverage, neither of which build can emit.
        position_owner: dict[str, dict[int, str]] = {
            doc_id: {} for doc_id in doc_ids}

        prev_term = None
        for entry in terms:
            if (not isinstance(entry, dict)
                    or not isinstance(entry.get("term"), str)
                    or not isinstance(entry.get("postings"), list)):
                raise DataError("snapshot: malformed term entry")
            term = entry["term"]
            if term == "":
                raise DataError("snapshot: term must be non-empty")
            if prev_term is not None and not term > prev_term:
                raise DataError(
                    f"snapshot: terms must be unique and strictly increasing "
                    f"({term!r} after {prev_term!r})")
            # A v1 term is exactly what the current analyzer would produce
            # from it: one normalized token identical to the term itself.
            tokens = tokenize(term)
            if tokens != [term]:
                raise DataError(
                    f"snapshot: term {term!r} is not a single normalized "
                    f"analyzer term")
            prev_term = term

            if not entry["postings"]:
                raise DataError(
                    f"snapshot: term {term!r} has an empty postings list")

            term_postings: dict[str, list[int]] = {}
            prev_posting_id = None
            for p in entry["postings"]:
                if (not isinstance(p, dict)
                        or not isinstance(p.get("id"), str)
                        or not isinstance(p.get("positions"), list)):
                    raise DataError("snapshot: malformed posting")
                doc_id = p["id"]
                positions = p["positions"]
                if doc_id not in seen:
                    raise DataError(
                        f"snapshot: posting references unknown document "
                        f"{doc_id!r}")
                if prev_posting_id is not None and not doc_id > prev_posting_id:
                    raise DataError(
                        f"snapshot: postings for {term!r} must be unique and "
                        f"strictly increasing by document id ({doc_id!r} "
                        f"after {prev_posting_id!r})")
                prev_posting_id = doc_id

                if not positions:
                    raise DataError(
                        f"snapshot: empty positions for "
                        f"{term!r}/{doc_id!r}")
                prev_pos = None
                for x in positions:
                    if not isinstance(x, int) or isinstance(x, bool) or x < 0:
                        raise DataError(
                            "snapshot: positions must be non-negative "
                            "integers (not booleans)")
                    if prev_pos is not None and x <= prev_pos:
                        raise DataError(
                            f"snapshot: positions for {term!r}/{doc_id!r} "
                            f"must be strictly increasing")
                    prev_pos = x
                    owners = position_owner[doc_id]
                    other = owners.get(x)
                    if other is not None:
                        raise DataError(
                            f"snapshot: position {x} in document {doc_id!r} "
                            f"claimed by both {other!r} and {term!r}")
                    owners[x] = term

                term_postings[doc_id] = positions
            postings[term] = term_postings

        # Cross-term coverage per document: a tokenized document's positions
        # are exactly 0..n-1, each owned by one term, with no gaps; a document
        # with zero tokens must never be referenced by a posting.
        for doc_id in doc_ids:
            owners = position_owner[doc_id]
            if not owners:
                # No posting references this document: it is an empty-text
                # document, which is the only build output with zero tokens.
                continue
            # Positions are distinct non-negative ints all <= max, so a count
            # of max+1 holds exactly when they fill the whole run 0..max.
            max_pos = max(owners)
            if len(owners) != max_pos + 1:
                raise DataError(
                    f"snapshot: positions for document {doc_id!r} must form a "
                    f"continuous run 0..{max_pos} "
                    f"(got {len(owners)} position(s))")

        return cls(list(doc_ids), postings)
