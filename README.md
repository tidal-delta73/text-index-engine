# text-index-engine

Full-text inverted index engine.

Pure-Python, no runtime dependencies. Requires Python 3.10+.

## Usage

```bash
python3 -m text_index_engine version
python3 -m text_index_engine help
python3 -m text_index_engine build <docs.jsonl> <snapshot>
python3 -m text_index_engine search <snapshot> <query>
```

### build

Reads a UTF-8 JSON Lines document set (one object per line with a non-empty,
unique string `id` and a string `text`) and writes a versioned positional
inverted-index snapshot. Text is normalized with Unicode NFKC + casefold and
tokenized into runs of letters/digits; positions are zero-based per document.
The snapshot is canonical JSON: documents, terms and postings are sorted by
Unicode code point and no environment data is stored, so the same document
set produces byte-identical output regardless of input line order. The target
file is replaced atomically and only after the snapshot is fully built.

### search

Reads only the snapshot and prints one JSON array of matching document ids,
sorted by Unicode code point. Query syntax: terms, `"double-quoted phrases"`
(terms must be positionally consecutive), explicit uppercase `AND`, `OR`,
`NOT` and parentheses; precedence is `NOT` > `AND` > `OR`, with no implicit
operators. Query terms use the same normalization and tokenization as the
index; a bare word that tokenizes to several terms is matched like a phrase.
`NOT` is relative to all documents in the snapshot. An empty query prints
`[]`.

### Exit codes

- `0` — success
- `1` — input file missing/unreadable, or output not writable
- `2` — data or query errors (malformed JSON lines, missing/wrong-typed
  fields, empty or duplicate ids, unsupported snapshot version, unknown
  query tokens, missing operands, unpaired parentheses/quotes, bad usage);
  stderr's first line starts with `error: ` and no partial results are
  produced
