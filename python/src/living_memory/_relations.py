"""How a value becomes keyed sheets."""

__all__ = [
    "Key",
    "Definition",
    "Variants",
    "assemble",
    "node_sheets",
    "node_rows",
    "line_rows",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from living_memory import _fields, _json
from living_memory._json_pointer import pointer

# A lines sheet and a node sheet have these columns (spec › item.2,
# line.1, node.1).
_LINES = ["address", "pointer", "line", "value"]
_NODE = ["address", "pointer", "type", "value"]

_Row = tuple[str, list[str]]
_Sheet = tuple[str, list[str]]


@dataclass(frozen=True)
class Key:
    # A row is keyed by its address and a pointer; a child's key extends
    # its parent's last field, so the parent's key is carried in it
    # (spec › item.2, attributes.1, nested.1).
    fields: tuple[str, ...]

    def extend(self, token: str | int) -> "Key":
        return Key((*self.fields[:-1], pointer(self.fields[-1], token)))


@dataclass(frozen=True)
class Definition:
    # In a schema definition, simple properties are columns, a text
    # column has a lines sheet, a value of any shape is a node sheet, a
    # nested object is a child sheet, and parts or tools named by their
    # type value are variants (spec › field.1, line.1, node.1, nested.1,
    # item.1, node.2).
    columns: tuple[str, ...] = ()
    lines: frozenset[str] = frozenset()
    nodes: tuple[str, ...] = ()
    children: tuple[tuple[str, "Definition"], ...] = ()
    variants: tuple[tuple[str, "Variants"], ...] = ()
    properties: frozenset[str] = field(default_factory=frozenset)

    def sheets(self, sheet: str) -> list[_Sheet]:
        # A sheet is followed by its lines sheets, then by its child
        # sheets (spec › file.4).
        sheets = [(sheet, ["address", "pointer", *self.columns])]
        for name in self.columns:
            if name in self.lines:
                sheets.append((f"{sheet}.{name}", _LINES))
        for name in self.nodes:
            sheets += node_sheets(f"{sheet}.{name}")
        for name, child in self.children:
            sheets += child.sheets(f"{sheet}.{name}")
        sheets += node_sheets(f"{sheet}.additionalProperties")
        for name, variants in self.variants:
            sheets += variants.sheets(f"{sheet}.{name}")
        return sheets

    def array_rows(self, sheet: str, key: Key, value: Any) -> list[_Row]:
        # An array keeps its order in the pointer (spec › nested.2).
        rows: list[_Row] = []
        for index, item in enumerate(_json.array(value)):
            rows += self.item_rows(sheet, key.extend(index), item)
        return rows

    def item_rows(self, sheet: str, key: Key, value: Any) -> list[_Row]:
        # A property beyond the definition goes to additionalProperties
        # (spec › nested.1, node.1, node.2).
        if not isinstance(value, dict):
            return []
        row = [*key.fields] + [_fields.text(value.get(name)) for name in self.columns]
        rows: list[_Row] = [(sheet, row)]
        for name in self.columns:
            if name in self.lines:
                rows += line_rows(f"{sheet}.{name}", key.extend(name), value.get(name))
        for name in self.nodes:
            if name in value:
                rows += node_rows(f"{sheet}.{name}", key.extend(name), value[name])
        for name, child in self.children:
            rows += child.item_rows(
                f"{sheet}.{name}", key.extend(name), value.get(name)
            )
        for name, member in value.items():
            if name not in self.properties:
                rows += node_rows(
                    f"{sheet}.additionalProperties", key.extend(name), member
                )
        for name, variants in self.variants:
            rows += variants.array_rows(
                f"{sheet}.{name}", key.extend(name), value.get(name)
            )
        return rows


@dataclass(frozen=True)
class Variants:
    # A part or tool belongs to the definition its type value names only
    # when it validates against it; any other is generic
    # (spec › item.1).
    definitions: dict[str, Definition]
    belongs: Callable[[str, Any], bool]

    def sheets(self, sheet: str) -> list[_Sheet]:
        sheets: list[_Sheet] = []
        for name, definition in self.definitions.items():
            sheets += definition.sheets(f"{sheet}.{name}")
        return sheets

    def array_rows(self, sheet: str, key: Key, value: Any) -> list[_Row]:
        rows: list[_Row] = []
        for index, item in enumerate(_json.array(value)):
            name = self._named(item)
            rows += self.definitions[name].item_rows(
                f"{sheet}.{name}", key.extend(index), item
            )
        return rows

    def _named(self, item: Any) -> str:
        name = item.get("type") if isinstance(item, dict) else ""
        named = isinstance(name, str) and name in self.definitions
        return name if named and self.belongs(name, item) else "generic"


def assemble(
    headers: list[_Sheet], rows: list[_Row]
) -> tuple[list[dict[str, Any]], set[str]]:
    # Every sheet is written, in order, each with its header; a field
    # MTSV cannot hold is left empty and reported, unless its line
    # breaks are carried by a lines sheet (spec › file.4, character.1,
    # line.1).
    names = {name for name, _ in headers}
    header_of = dict(headers)
    records: dict[str, list[list[str]]] = {name: [] for name, _ in headers}
    left: set[str] = set()
    for name, row in rows:
        fields = []
        for column, value in zip(header_of[name], row):
            written = _fields.field(value)
            carried = "\n" in value and f"{name}.{column}" in names
            if written != value and not carried:
                left.add(f"{name}.{column}")
            fields.append(written)
        records[name].append(fields)
    sheets = [
        {"sheet name": name, "header": header, "records": records[name]}
        for name, header in headers
    ]
    return sheets, left


def node_sheets(sheet: str) -> list[_Sheet]:
    # A node sheet is followed by the lines sheet of its value column
    # (spec › line.1).
    return [(sheet, _NODE), (f"{sheet}.value", _LINES)]


def node_rows(sheet: str, key: Key, value: Any) -> list[_Row]:
    # Each node is a row, a node before the nodes within it
    # (spec › node.1).
    rows: list[_Row] = [(sheet, [*key.fields, _json.type(value), _fields.text(value)])]
    rows += line_rows(f"{sheet}.value", key, value)
    if isinstance(value, dict):
        for name, member in value.items():
            rows += node_rows(sheet, key.extend(name), member)
    elif isinstance(value, list):
        for index, element in enumerate(value):
            rows += node_rows(sheet, key.extend(index), element)
    return rows


def line_rows(sheet: str, key: Key, value: Any) -> list[_Row]:
    # Each line is a row of the lines sheet, keyed by its position
    # (spec › line.1).
    return [
        (sheet, [*key.fields, str(index), line])
        for index, line in enumerate(_fields.lines(value))
    ]
