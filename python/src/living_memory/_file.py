"""How the whole input is written as one MTSV file."""

__all__ = ["names", "write"]

from functools import partial

import mtsv

from living_memory import _order, _record, _relation, _sheet
from living_memory._relation import Layout, Placed, Relation


def write(sheet_layout: Layout, placed: list[Placed]) -> str:
    # One file holds every sheet the schema gives that holds a record,
    # in its place, written by the MTSV generator (Wickham, 3.5. One
    # type in multiple tables; MTSV draft, Generators; spec › file.1-3).
    paths = _paths(sheet_layout)
    held = _order.records([path[-1] for path in paths], placed, _relation.relation)
    return mtsv.dumps(
        [
            {
                "sheet name": _sheet.name(path),
                "header": _sheet.header(path[-1]),
                "records": [_record.fields(one) for one in records],
            }
            for path, records in zip(paths, held)
            if records
        ]
    )


def names(sheet_layout: Layout) -> list[str]:
    # The name of every sheet the schema gives, in the file's order,
    # by which the parts of an input are appended (spec › value.2,
    # order.1).
    return [_sheet.name(path) for path in _paths(sheet_layout)]


def _paths(sheet_layout: Layout) -> list[tuple[Relation, ...]]:
    return _order.sheets(
        _relation.root(sheet_layout),
        partial(_relation.children, sheet_layout),
        _relation.last(sheet_layout),
    )
