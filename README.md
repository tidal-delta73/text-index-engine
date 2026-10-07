# text-index-engine

Full-text inverted index engine.

Pure-Python, no runtime dependencies. Requires Python 3.10+.

## Commands

```bash
python3 -m text_index_engine version
python3 -m text_index_engine help
python3 -m text_index_engine build <input.jsonl> <snapshot>
python3 -m text_index_engine search <snapshot> "<query>"
python3 -m text_index_engine rank <snapshot> "<query>"
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
| `foo*`        | prefix: every dictionary term starting with `foo`    |
| `"new york"`  | phrase: all terms consecutive in one document        |
| `AND OR NOT`  | boolean operators (uppercase only)                   |
| `( ... )`     | grouping                                             |

A prefix is written as a bare term immediately followed by one `*` (e.g.
`app*`). The text before the star is normalized/tokenized like an ordinary
term and must analyze to exactly one non-empty term; every snapshot term
starting with that normalized string contributes its documents. Matching is a
codepoint-by-codepoint prefix on normalized Unicode strings — no extra
tokenization, stemming or locale collation, and a prefix with no matching
term matches nothing. A lone `*`, more than one `*`, or a `*` anywhere but
the end of a bare term is a query error; `AND*`/`OR*`/`NOT*` are prefix
terms, while bare uppercase `AND`/`OR`/`NOT` stay operators. Inside quoted
phrases `*` has no wildcard meaning.

`NOT`'s universe is every document in the snapshot. An empty query prints
`[]`. Search reads only the snapshot — never the original documents — and
does no relevance scoring.

Examples:

```bash
python3 -m text_index_engine search idx.snap 'fox AND NOT "lazy dog"'
python3 -m text_index_engine search idx.snap '(quick OR fast) AND fox'
python3 -m text_index_engine search idx.snap 'NOT app* OR "new york"'
```

### rank

Same snapshot, query language and candidate set as `search`, but the
matching documents are scored with BM25 and printed as a compact JSON array
of `{"id": ..., "score": "..."}` objects — score descending, ties broken by
document id in code point order, one trailing newline. The score is a fixed
six-decimal string (rounded half to even; never NaN, Infinity or negative
zero). An empty query or empty candidate set prints `[]`.

Only the query tree's *positive* leaves (those under an even number of
`NOT`s) score; leaves under an odd `NOT` count still filter candidates but
contribute nothing — so `NOT fox` ranks its candidates with all-zero scores
in id order. A term contributes itself, a phrase contributes each of its
words (no phrase bonus), a prefix contributes every dictionary term it
expands to, and repeated occurrences of a term accumulate. With `N` the
snapshot's document count, `dl` a document's position total and `avgdl` the
mean position total over all documents (empty ones included):

```
idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
k1 = 1.2, b = 0.75
```

`avgdl == 0` scores every candidate `0.000000`. Ranking reads only the
snapshot — never the original documents — and any v1 snapshot serves it
without a rebuild.

## Exit codes

| Code | Meaning                                                                   |
|------|---------------------------------------------------------------------------|
| 0    | success                                                                   |
| 1    | filesystem error (missing/unreadable input, unwritable output)            |
| 2    | invalid data: malformed document/query, unsupported snapshot version, etc. |

On failure the first stderr line starts with `error:` and no partial result
is emitted.
