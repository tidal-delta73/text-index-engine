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

Takes the same query language and the same snapshot as `search`, and prints a
compact JSON array of the boolean query's candidate documents, each as
`{"id": ..., "score": ...}` with a fixed six-decimal decimal string. Results
are ordered by score descending; equal scores tie-break by document id in
Unicode code point order. An empty query or an empty candidate set prints
`[]`. Only the snapshot is read.

Scoring uses BM25 with k1 = 1.2 and b = 0.75:

```
idf   = ln(1 + (N - df + 0.5) / (df + 0.5))
score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
```

N counts every document in the snapshot (including empty ones), dl is the
document's total token count, and avgdl averages over every document. Only
positive leaves — leaves under an even number of `NOT`s — contribute. A term
leaf contributes its term, a phrase leaf contributes each phrase word (no
phrase bonus), and a prefix leaf contributes every dictionary term it expands
to; repeated occurrences accumulate. Leaves under an odd number of `NOT`s
still filter candidates but never score, so a valid query without positive
leaves (e.g. `NOT fox`) returns its hits at score `0.000000`.

```bash
python3 -m text_index_engine rank idx.snap '(quick OR fast) AND fox'
```

## Exit codes

| Code | Meaning                                                                   |
|------|---------------------------------------------------------------------------|
| 0    | success                                                                   |
| 1    | filesystem error (missing/unreadable input, unwritable output)            |
| 2    | invalid data: malformed document/query, unsupported snapshot version, etc. |

On failure the first stderr line starts with `error:` and no partial result
is emitted.
