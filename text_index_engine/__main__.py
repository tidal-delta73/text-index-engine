"""Command line entry point.

Commands:
  version                          print the package version
  help                             print this message
  build <input.jsonl> <snapshot>   build a deterministic inverted snapshot
  search <snapshot> "<query>"      run one boolean query against a snapshot
  rank <snapshot> "<query>"        BM25-rank the query's candidate documents

Exit codes: 0 success, 1 filesystem error (missing/unreadable input,
unwritable output), 2 invalid data (documents, snapshot or query). On error
the first stderr line starts with "error:" and no partial result is written
to stdout.
"""
import json
import sys

from . import __version__
from .errors import DataError
from .query import evaluate, parse
from .rank import rank
from .snapshot import Snapshot, build_from_lines, write_atomic

USAGE = """usage: python3 -m text_index_engine <command>

commands:
  version                                 print the package version
  help                                    print this message
  build <input.jsonl> <snapshot>          build an inverted-index snapshot
  search <snapshot> "<query>"             query a snapshot
  rank <snapshot> "<query>"               BM25-rank the query's candidates

exit codes:
  0  success
  1  filesystem error (missing/unreadable input, unwritable output)
  2  invalid data (documents, snapshot or query)
"""


def _data_error(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


def _io_error(exc: OSError) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


def _cmd_build(args: list[str]) -> int:
    if len(args) != 2:
        print("error: build requires <input.jsonl> and <snapshot> paths",
              file=sys.stderr)
        return 2
    input_path, output_path = args
    try:
        # The file object itself is the one-shot line iterable; lines are
        # consumed as the index is built, not buffered as a whole first.
        with open(input_path, "r", encoding="utf-8") as fp:
            data = build_from_lines(fp)
    except UnicodeDecodeError as exc:
        return _data_error(DataError(f"input is not valid UTF-8: {exc}"))
    except OSError as exc:
        return _io_error(exc)
    except DataError as exc:
        return _data_error(exc)

    try:
        write_atomic(output_path, data)
    except OSError as exc:
        return _io_error(exc)
    return 0


def _cmd_search(args: list[str]) -> int:
    if len(args) != 2:
        print("error: search requires <snapshot> path and one query string",
              file=sys.stderr)
        return 2
    snapshot_path, query = args
    try:
        with open(snapshot_path, "rb") as fp:
            raw = fp.read()
    except OSError as exc:
        return _io_error(exc)

    try:
        snapshot = Snapshot.load(raw)
        node = parse(query)
        hits = set() if node is None else evaluate(node, snapshot)
        result = sorted(hits)
        output = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    except DataError as exc:
        return _data_error(exc)

    sys.stdout.write(output + "\n")
    return 0


def _cmd_rank(args: list[str]) -> int:
    if len(args) != 2:
        print("error: rank requires <snapshot> path and one query string",
              file=sys.stderr)
        return 2
    snapshot_path, query = args
    try:
        with open(snapshot_path, "rb") as fp:
            raw = fp.read()
    except OSError as exc:
        return _io_error(exc)

    try:
        snapshot = Snapshot.load(raw)
        node = parse(query)
        result = [] if node is None else rank(node, snapshot)
        output = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    except DataError as exc:
        return _data_error(exc)

    sys.stdout.write(output + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    if command == "build":
        return _cmd_build(args[1:])
    if command == "search":
        return _cmd_search(args[1:])
    if command == "rank":
        return _cmd_rank(args[1:])
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
