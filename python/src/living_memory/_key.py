"""How each record is identified: its primary key."""

__all__ = [
    "PARENT",
    "POINTER",
    "ROOT",
    "RUN",
    "SUBORDINATE",
    "member",
    "root",
    "run",
    "values",
]

from living_memory import _json_pointer

# A subordinate record's first key is the pointer of the record whose
# instance holds its own, copied down, then its own pointer; the sheet
# of the input values is keyed by pointer alone, and a run by its
# string's pointer, then its page, line and position (Codd, 1.4. Normal
# Form; spec › key.1, key.2, key.3, key.5).
PARENT = "parent"
POINTER = "pointer"
ROOT = (POINTER,)
SUBORDINATE = (PARENT, POINTER)
RUN = (POINTER, "page", "line", "position")


def root(position: int) -> str:
    # An input value's pointer has its zero-based position in the input
    # as the first reference token (spec › key.2).
    return _json_pointer.pointer("", position)


def member(at: str, token: str | int) -> str:
    # A member or element is identified by its JSON Pointer, '~' written
    # '~0' and '/' written '~1' (RFC 6901, 3. Syntax; spec › key.3,
    # key.4).
    return _json_pointer.pointer(at, token)


def values(columns: tuple[str, ...], parent: str | None, at: str) -> dict[str, str]:
    # A record's key values: its parent's, where its sheet has that
    # column, and its own pointer (spec › key.2, key.3).
    return {c: v for c, v in ((PARENT, parent), (POINTER, at)) if c in columns}


def run(at: str, counts: tuple[int, int, int]) -> dict[str, str]:
    # A run's key values: its string's pointer, then its page, line and
    # position (spec › key.5).
    return dict(zip(RUN, (at, *map(str, counts))))
