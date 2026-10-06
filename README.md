# text-index-engine

Full-text inverted index engine.

Pure-Python, no runtime dependencies. Requires Python 3.10+.

## Commands

```bash
python3 -m text_index_engine version
python3 -m text_index_engine help
python3 -m text_index_engine build <input.jsonl> <snapshot>
python3 -m text_index_engine search <snapshot> "<query>"
```

### build

Reads a UTF-8 [JSON Lines](https://jsonlines.org/) document set. Each line is
a JSON object with a non-empty, unique string `id` and a string `text`:

```json
{"id": "d1", "text": "The quick brown fox"}
{"id": "d2", "text": "快狐跳过懒狗"}
{"id": "d3", "text": ""}
```

Text is NFKC-normalized and casefolded before tokenization; tokens are
maximal runs of Unicode letters or numbers, with zero-based positions per
document. Documents with empty text are still indexed (so `NOT` covers them).

The snapshot is canonical versioned JSON: documents, terms and postings are
ordered by Unicode code point (document id / normalized term), and no
environment data (timestamps, paths) is written. The same document set
produces a byte-identical snapshot regardless of input line order. The
output file is replaced atomically, only after the whole input validates.

### search

Takes one query string and prints a JSON array of matching document ids,
sorted by Unicode code point. Operators are explicit uppercase keywords with
precedence `NOT` > `AND` > `OR`; there is no implicit operator.

| Syntax        | Meaning                                              |
|---------------|------------------------------------------------------|
| `foo`         | term (normalized/tokenized like indexed text)        |
| `app*`        | term prefix: all terms starting with `app`           |
| `"new york"`  | phrase: all terms consecutive in one document        |
| `AND OR NOT`  | boolean operators (uppercase only)                   |
| `( ... )`     | grouping                                             |

In `app*` the text before the star is analyzed with the same NFKC
normalization and casefolding as a plain term; it must analyze to exactly
one non-empty token. The query then matches every dictionary term with that
normalized token as a code-point prefix (no extra tokenization, stemming or
locale comparison), and a prefix with no matching terms matches nothing.
The star is meaningful only at the end of a bare term — `*` alone, several
stars, a star before the term's end, or a prefix body that tokenizes to
zero or several terms is a query error; inside quoted phrases `*` is
literal. `AND*`, `OR*` and `NOT*` are prefix terms, while bare `AND`, `OR`
and `NOT` remain operators.

`NOT`'s universe is every document in the snapshot. An empty query prints
`[]`. Search reads only the snapshot — never the original documents — and
does no relevance scoring.

Examples:

```bash
python3 -m text_index_engine search idx.snap 'fox AND NOT "lazy dog"'
python3 -m text_index_engine search idx.snap '(quick OR fast) AND fox'
python3 -m text_index_engine search idx.snap 'bro* OR app*'
```

## Exit codes

| Code | Meaning                                                                   |
|------|---------------------------------------------------------------------------|
| 0    | success                                                                   |
| 1    | filesystem error (missing/unreadable input, unwritable output)            |
| 2    | invalid data: malformed document/query, unsupported snapshot version, etc. |

On failure the first stderr line starts with `error:` and no partial result
is emitted.
