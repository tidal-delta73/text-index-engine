"""Command line entry point: version, help, build and search."""
import json
import os
import sys
import tempfile

from . import __version__
from .query import QueryError, evaluate, lex, parse
from .snapshot import SnapshotError, build_snapshot_bytes, load_snapshot, parse_documents

USAGE = """usage: python3 -m text_index_engine <command>

commands:
  version                    print the package version
  help                       print this message
  build <docs> <snapshot>    build an index snapshot from a JSON Lines document set
  search <snapshot> <query>  search a snapshot and print matching document ids
"""


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _build(args: list[str]) -> int:
    if len(args) != 2:
        return _fail("usage: build <docs> <snapshot>", 2)
    src, dst = args
    try:
        with open(src, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return _fail(f"cannot read {src}: {exc.strerror or exc}", 1)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _fail(f"{src} is not valid UTF-8", 1)
    try:
        docs = parse_documents(text)
        payload = build_snapshot_bytes(docs)
    except SnapshotError as exc:
        return _fail(str(exc), 2)
    # Replace the target only after the snapshot is fully built and written.
    directory = os.path.dirname(os.path.abspath(dst))
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".snapshot-", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dst)
        tmp_path = None
    except OSError as exc:
        return _fail(f"cannot write {dst}: {exc.strerror or exc}", 1)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return 0


def _search(args: list[str]) -> int:
    if len(args) != 2:
        return _fail("usage: search <snapshot> <query>", 2)
    src, query_text = args
    try:
        with open(src, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return _fail(f"cannot read {src}: {exc.strerror or exc}", 1)
    try:
        snapshot = load_snapshot(raw)
        ast = parse(lex(query_text))
    except (SnapshotError, QueryError) as exc:
        return _fail(str(exc), 2)
    ids = [] if ast is None else evaluate(ast, snapshot)
    sys.stdout.write(json.dumps(ids, ensure_ascii=False, separators=(",", ":")) + "\n")
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
        return _build(args[1:])
    if command == "search":
        return _search(args[1:])
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
