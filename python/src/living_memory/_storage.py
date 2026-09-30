"""What a data bank holds of an input: its values, each whole."""

__all__ = ["Stored", "position", "whole"]

from typing import Any

from living_memory import _key

# An input's stored sheets, each with its place among every sheet the
# schema gives (spec › Fields, _place).
Stored = list[tuple[int, dict[str, Any]]]


def position(sheet: dict[str, Any], record: list[str]) -> int:
    # A record's value is at the position its key _input value gives,
    # whatever the sheet (spec › key.2, key.3, key.5).
    return int(record[sheet["header"].index(_key.INPUT_VALUE)])


def whole(stored: Stored, values: int) -> Stored:
    # Each sheet keeps only the records of the values stored whole,
    # those at positions below their number (spec › storage.4, key.2).
    kept: Stored = []
    for place, sheet in stored:
        records = [r for r in sheet["records"] if position(sheet, r) < values]
        kept.append((place, {**sheet, "records": records}))
    return kept
