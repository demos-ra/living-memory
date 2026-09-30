"""What a data bank holds of an input: its values, each whole."""

__all__ = ["Stored", "count", "whole"]

from typing import Any

from living_memory import _key

# An input's stored sheets, each with its place among every sheet the
# schema gives (spec › Fields, place).
Stored = list[tuple[int, dict[str, Any]]]


def count(stored: Stored) -> int:
    # The number of values stored: the records of the sheet of the input
    # values, at place 0, whose pointer is a value's own, one reference
    # token (spec › storage.3, storage.5, key.2, order.1).
    for place, sheet in stored:
        if place == 0:
            at = sheet["header"].index(_key.POINTER)
            return sum(1 for record in sheet["records"] if _is_value(record[at]))
    return 0


def whole(stored: Stored, values: int) -> Stored:
    # Each sheet keeps only the records of the values stored whole,
    # those at positions below their number (spec › storage.4, key.2).
    kept: Stored = []
    for place, sheet in stored:
        at = sheet["header"].index(_key.POINTER)
        records = [r for r in sheet["records"] if _key.position(r[at]) < values]
        kept.append((place, {**sheet, "records": records}))
    return kept


def _is_value(pointer: str) -> bool:
    # An input value's own pointer holds one reference token (spec ›
    # key.2).
    return pointer.count("/") == 1
