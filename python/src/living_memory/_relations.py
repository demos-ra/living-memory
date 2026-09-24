"""How a value becomes keyed sheets."""

__all__ = [
    "Key",
    "Definition",
    "Variants",
    "assemble",
    "node_sheets",
    "node_rows",
    "line_rows",
    "text",
    "pointer",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from living_memory import _json

# spec › item.2, line.1.
_LINES = ["address", "pointer", "line", "value"]
# spec › node.1.
_NODE = ["address", "pointer", "type", "value"]
# The draft, Generators.
_SEPARATORS = frozenset("\t\n\f\r")

_Row = tuple[str, list[str]]
_Sheet = tuple[str, list[str]]


@dataclass(frozen=True)
class Key:
    # spec › item.2, attributes.1, nested.1.
    fields: tuple[str, ...]

    def extend(self, token: str | int) -> "Key":
        return Key((*self.fields[:-1], pointer(self.fields[-1], token)))


@dataclass(frozen=True)
class Definition:
    # spec › field.1, line.1, node.1, nested.1, item.1, node.2.
    columns: tuple[str, ...] = ()
    lines: frozenset[str] = frozenset()
    nodes: tuple[str, ...] = ()
    children: tuple[tuple[str, "Definition"], ...] = ()
    variants: tuple[tuple[str, "Variants"], ...] = ()
    properties: frozenset[str] = field(default_factory=frozenset)

    def sheets(self, sheet: str) -> list[_Sheet]:
        # spec › file.4.
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
        # spec › nested.2.
        rows: list[_Row] = []
        for index, item in enumerate(_list(value)):
            rows += self.item_rows(sheet, key.extend(index), item)
        return rows

    def item_rows(self, sheet: str, key: Key, value: Any) -> list[_Row]:
        # spec › nested.1, node.1, node.2.
        if not isinstance(value, dict):
            return []
        row = [*key.fields] + [text(value.get(name)) for name in self.columns]
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
    # spec › item.1.
    definitions: dict[str, Definition]
    belongs: Callable[[str, Any], bool]

    def sheets(self, sheet: str) -> list[_Sheet]:
        sheets: list[_Sheet] = []
        for name, definition in self.definitions.items():
            sheets += definition.sheets(f"{sheet}.{name}")
        return sheets

    def array_rows(self, sheet: str, key: Key, value: Any) -> list[_Row]:
        rows: list[_Row] = []
        for index, item in enumerate(_list(value)):
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
    # spec › file.4, character.1, line.1.
    names = {name for name, _ in headers}
    header_of = dict(headers)
    records: dict[str, list[list[str]]] = {name: [] for name, _ in headers}
    left: set[str] = set()
    for name, row in rows:
        fields = []
        for column, value in zip(header_of[name], row):
            written = _write_cell(value)
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
    # spec › line.1.
    return [(sheet, _NODE), (f"{sheet}.value", _LINES)]


def node_rows(sheet: str, key: Key, value: Any) -> list[_Row]:
    # spec › node.1.
    rows: list[_Row] = [(sheet, [*key.fields, _json.type(value), text(value)])]
    rows += line_rows(f"{sheet}.value", key, value)
    if isinstance(value, dict):
        for name, member in value.items():
            rows += node_rows(sheet, key.extend(name), member)
    elif isinstance(value, list):
        for index, element in enumerate(value):
            rows += node_rows(sheet, key.extend(index), element)
    return rows


def line_rows(sheet: str, key: Key, value: Any) -> list[_Row]:
    # spec › line.1.
    return [
        (sheet, [*key.fields, str(index), line])
        for index, line in enumerate(_split_lines(text(value)))
    ]


def text(value: Any) -> str:
    # spec › node.1, text.1; RFC 8259, Section 3.
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else ""


def pointer(at: str, token: str | int) -> str:
    # RFC 6901, Section 3.
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{at}/{escaped}"


def _write_cell(value: str) -> str:
    # spec › character.1.
    return "" if not _SEPARATORS.isdisjoint(value) else value


def _split_lines(value: str) -> list[str]:
    # spec › line.1.
    if "\n" not in value:
        return []
    found = value.split("\n")
    return [line.removesuffix("\r") for line in found[:-1]] + found[-1:]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
