"""What a data bank communicates: the names of what it stores, and its
records by their keys."""

__all__ = ["names", "records"]

import mtsv

from living_memory import _key, _storage
from living_memory._storage import Stored

# The sheets of the names, and their columns, keys first (spec ›
# communication.3, Fields).
_INPUTS = ("inputs", ["input", "values"])
_SHEETS = ("sheets", ["input", "place", "sheet name"])
_FIELDS = ("fields", ["input", "place", "position", "field name"])


def names(inputs: list[tuple[str, Stored]]) -> str:
    # The names of what is stored, as MTSV: each input with the number
    # of its values, each sheet by its place and name, and each field of
    # its header by its position, counted from 0 (Codd, 1.3. A
    # Relational View of Data; spec › communication.1, communication.3).
    held = {"inputs": [], "sheets": [], "fields": []}
    for name, stored in inputs:
        held["inputs"].append([name, str(_storage.count(stored))])
        for place, sheet in stored:
            held["sheets"].append([name, str(place), sheet["sheet name"]])
            held["fields"] += [
                [name, str(place), str(position), field]
                for position, field in enumerate(sheet["header"])
            ]
    return mtsv.dumps(
        [
            {"sheet name": sheet_name, "header": header, "records": held[sheet_name]}
            for sheet_name, header in (_INPUTS, _SHEETS, _FIELDS)
        ]
    )


def records(
    stored: Stored, first: int, last: int, places: list[int] | None = None
) -> str:
    # The records of the values at positions first to last, both
    # included, of the sheets at the places asked for, else of every
    # sheet, each in its stored sheet, in its place; a sheet holding
    # none of them is left out; values as stored and records in stored
    # order (spec › communication.1, communication.2, communication.4).
    found = []
    for place, sheet in stored:
        if places is not None and place not in places:
            continue
        at = sheet["header"].index(_key.POINTER)
        kept = [r for r in sheet["records"] if first <= _key.position(r[at]) <= last]
        if kept:
            found.append({**sheet, "records": kept})
    return mtsv.dumps(found)
