"""Error classification mapping to CLI exit codes.

DataError covers invalid user-supplied content (documents, snapshots,
queries): the command exits with status 2. Filesystem problems are raised as
the built-in OSError family and map to status 1 in the CLI.
"""


class DataError(Exception):
    """Invalid input data or query; the command exits with status 2."""
