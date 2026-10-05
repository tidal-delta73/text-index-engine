"""Snapshot format: build a canonical positional inverted index and load it.

The snapshot is a canonical JSON document with a fixed version. Documents,
terms and postings are ordered by Unicode code point of the raw document id
or normalized term, and no environment data (timestamps, paths) is stored,
so the same document set always produces byte-identical UTF-8 output.
"""
import json

from .tokenizer import tokenize

SNAPSHOT_VERSION = 1


class SnapshotError(Exception):
    """A document-set or snapshot data error (reported with exit code 2)."""


def parse_documents(text: str) -> list[tuple[str, str]]:
    """Parse and validate a UTF-8 JSON Lines document set.

    Each line must be a JSON object with a non-empty unique string ``id``
    and a string ``text``. Returns a list of ``(id, text)`` pairs.
    """
    docs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise SnapshotError(f"line {lineno}: not valid JSON") from None
        if not isinstance(obj, dict):
            raise SnapshotError(f"line {lineno}: not a JSON object")
        doc_id = obj.get("id")
        if not isinstance(doc_id, str):
            raise SnapshotError(f"line {lineno}: 'id' is missing or not a string")
        if not doc_id:
            raise SnapshotError(f"line {lineno}: 'id' is empty")
        if doc_id in seen:
            raise SnapshotError(f"line {lineno}: duplicate id {doc_id!r}")
        doc_text = obj.get("text")
        if not isinstance(doc_text, str):
            raise SnapshotError(f"line {lineno}: 'text' is missing or not a string")
        seen.add(doc_id)
        docs.append((doc_id, doc_text))
    return docs


def build_snapshot_bytes(docs: list[tuple[str, str]]) -> bytes:
    """Serialize the positional inverted index for ``docs`` to canonical JSON."""
    doc_ids = sorted(doc_id for doc_id, _ in docs)
    index_of = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    terms: dict[str, dict[int, list[int]]] = {}
    for doc_id, doc_text in docs:
        doc_index = index_of[doc_id]
        for position, term in enumerate(tokenize(doc_text)):
            postings = terms.setdefault(term, {})
            postings.setdefault(doc_index, []).append(position)
    terms_obj = {
        term: [[doc_index, positions] for doc_index, positions in sorted(postings.items())]
        for term, postings in terms.items()
    }
    snapshot = {"version": SNAPSHOT_VERSION, "documents": doc_ids, "terms": terms_obj}
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode("utf-8")


class Snapshot:
    """A loaded snapshot: document ids plus term -> [(doc index, positions)]."""

    def __init__(self, documents: list[str], terms: dict[str, list[tuple[int, list[int]]]]):
        self.documents = documents
        self.terms = terms


def load_snapshot(data: bytes) -> Snapshot:
    """Parse and validate snapshot bytes."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SnapshotError("snapshot is not valid UTF-8") from None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        raise SnapshotError("snapshot is not valid JSON") from None
    if not isinstance(obj, dict):
        raise SnapshotError("snapshot is not a JSON object")
    version = obj.get("version")
    if not isinstance(version, int) or version != SNAPSHOT_VERSION:
        raise SnapshotError(f"unsupported snapshot version: {version!r}")
    documents = obj.get("documents")
    if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
        raise SnapshotError("snapshot 'documents' is not a list of strings")
    raw_terms = obj.get("terms")
    if not isinstance(raw_terms, dict):
        raise SnapshotError("snapshot 'terms' is not an object")
    terms: dict[str, list[tuple[int, list[int]]]] = {}
    for term, postings in raw_terms.items():
        if not isinstance(term, str) or not isinstance(postings, list):
            raise SnapshotError("snapshot 'terms' is malformed")
        checked: list[tuple[int, list[int]]] = []
        for posting in postings:
            if (
                not isinstance(posting, list)
                or len(posting) != 2
                or not isinstance(posting[0], int)
                or not 0 <= posting[0] < len(documents)
                or not isinstance(posting[1], list)
                or not all(isinstance(p, int) and p >= 0 for p in posting[1])
            ):
                raise SnapshotError(f"snapshot posting for term {term!r} is malformed")
            checked.append((posting[0], list(posting[1])))
        terms[term] = checked
    return Snapshot(list(documents), terms)
