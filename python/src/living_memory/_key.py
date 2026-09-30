"""How each record is identified: its primary key."""

__all__ = [
    "INPUT_VALUE",
    "ROOT",
    "RUN",
    "SUBORDINATE",
    "numbered",
    "position",
    "root",
    "run",
    "values",
]

from typing import Any

from living_memory import _json_pointer, _order

# The sheet of the input values is keyed by its input value's position;
# every other sheet by that position, its instance, its parent's
# instance and the pointer from its parent's instance; a run by its
# string's position and instance, then its page, line and position
# (Codd, 1.4. Normal Form; spec › key.2, key.3, key.5, Fields).
INPUT_VALUE = "_input value"
_INSTANCE = "_instance"
ROOT = (INPUT_VALUE,)
SUBORDINATE = (INPUT_VALUE, _INSTANCE, "_parent", "_pointer")
RUN = (INPUT_VALUE, _INSTANCE, "_page", "_line", "_position")


def root(position: int) -> str:
    # An input value's pointer within the input has its zero-based
    # position as its first reference token (spec › key.2, field.5).
    return _json_pointer.pointer("", position)


def position(at: str) -> int:
    # The position of the input value a pointer within the input is in.
    return int(_json_pointer.tokens(at)[0])


def numbered(value: Any, at: str) -> dict[str, int]:
    # Each instance of an input value, by its pointer within the input,
    # numbered from 0 in the order written, the value itself first (JSON
    # Schema, 4.2.1. Instance Data Model; spec › key.3, order.2).
    found: dict[str, int] = {}
    for number, (tokens, _) in enumerate(_order.instances(value)):
        where = at
        for token in tokens:
            where = _json_pointer.pointer(where, token)
        found[where] = number
    return found


def values(
    columns: tuple[str, ...],
    numbers: dict[str, int],
    at: str,
    parent: str | None,
    pointer: str,
) -> dict[str, str]:
    # A record's key values: its input value's position, then, where its
    # sheet has them, its instance, its parent's instance and the
    # pointer from it; a record of its location's own instance, or of an
    # input value, has no parent and no pointer (spec › key.2, key.3).
    found = {
        INPUT_VALUE: str(position(at)),
        _INSTANCE: str(numbers[at]),
        "_parent": "" if parent is None else str(numbers[parent]),
        "_pointer": "" if parent is None else pointer,
    }
    return {column: found[column] for column in columns}


def run(
    numbers: dict[str, int], at: str, counts: tuple[int, int, int]
) -> dict[str, str]:
    # A run's key values: its string's position and instance, then its
    # page, line and position (spec › key.5).
    return dict(zip(RUN, (str(position(at)), str(numbers[at]), *map(str, counts))))
