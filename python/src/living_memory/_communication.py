"""What a data bank communicates: the names of what it stores, and its
records by their keys."""

__all__ = ["names", "records"]

import mtsv

from living_memory import _storage
from living_memory._storage import Stored

# The sheets of the names, and their columns, keys first, each named as
# the specification fixes it (spec › communication.3, Fields).
_INPUTS = ("_inputs", ["_input", "_values"])
_SHEETS = ("_sheets", ["_input", "_place", "_sheet name"])
_FIELDS = ("_fields", ["_input", "_place", "_position", "_field name"])


def names(inputs: list[tuple[str, Stored, int]]) -> str:
    # The names of what is stored, as MTSV: each input with the number
    # of its values, each sheet by its place and name, and each field of
    # its header by its position, counted from 0 (Codd, 1.3. A
    # Relational View of Data; spec › communication.1, communication.3).
    held: dict[str, list[list[str]]] = {_INPUTS[0]: [], _SHEETS[0]: [], _FIELDS[0]: []}
    for name, stored, values in inputs:
        held[_INPUTS[0]].append([name, str(values)])
        for place, sheet in stored:
            held[_SHEETS[0]].append([name, str(place), sheet["sheet name"]])
            held[_FIELDS[0]] += [
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
    # none of them is left out, as the file of those values holds none
    # (spec › communication.1, communication.2, communication.4, value.2,
    # file.2).
    found = []
    for place, sheet in stored:
        if places is not None and place not in places:
            continue
        kept = [
            r for r in sheet["records"] if first <= _storage.position(sheet, r) <= last
        ]
        if kept:
            found.append({**sheet, "records": kept})
    return mtsv.dumps(found)
