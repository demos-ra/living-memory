"""How a value becomes keyed sheets.

Child sheets, keys, positions, values of any shape, numbers as written,
lines, values MTSV cannot represent not written.

Classes:
Number -- a JSON number, kept as the text the input writes
Shape -- the fields of a schema definition, and its child sheets

Functions:
assemble -- return every sheet, in order, each with its rows
shape_sheets -- return the sheets of a definition, in order
variant_sheets -- return the sheets of the definitions of an array
node_sheets -- return a node sheet and its lines sheet
array_entries -- return the rows of an array of one definition
variant_entries -- return the rows of an array of several definitions
item_entries -- return the rows of an object of a definition
node_entries -- return the rows of a value of any shape
text_entries -- return the lines of a text value
decode -- read a JSON text, keeping each number as written
kind -- return the type of a JSON value
text -- return the text of a string or a number
cell -- return a text as a field is written
lines -- return the lines of a text that holds a line break
pointer -- extend a JSON Pointer by one reference token

Constants:
LINES -- the header of a lines sheet of the eight
NODE -- the header of a node sheet of the eight
"""

__all__ = [
    "Number",
    "Shape",
    "assemble",
    "shape_sheets",
    "variant_sheets",
    "node_sheets",
    "array_entries",
    "variant_entries",
    "item_entries",
    "node_entries",
    "text_entries",
    "decode",
    "kind",
    "text",
    "cell",
    "lines",
    "pointer",
    "LINES",
    "NODE",
]

import json
from dataclasses import dataclass, field
from typing import Any

# spec › item.2, line.1: a row of the eight is keyed by address and
# pointer; a line adds its position.
LINES = ["address", "pointer", "line", "value"]
# spec › node.1: a row per node, its pointer, its type and its value.
NODE = ["address", "pointer", "type", "value"]

# The draft, Generators: a field cannot hold HT, LF, FF or CR.
_SEPARATORS = frozenset("\t\n\f\r")

Entry = tuple[str, list[str]]


class Number(str):
    """A JSON number, kept as the text the input writes.

    spec › node.3: an implementation may limit the range and precision
    of numbers, so a number is written as the input writes it.
    """


@dataclass(frozen=True)
class Shape:
    """The fields of a schema definition, and its child sheets.

    columns -- the simple fields, each a column (spec › field.1)
    lines -- the columns the schema types as string (spec › line.1)
    nodes -- the fields of any shape, each a node sheet (spec › node.1)
    children -- nested definitions, each a child sheet (spec › nested.1)
    variants -- arrays of definitions, named by type (spec › item.1)
    known -- every field of the definition; the rest go to its
        additionalProperties sheet (spec › node.2)
    """

    columns: tuple[str, ...] = ()
    lines: frozenset[str] = frozenset()
    nodes: tuple[str, ...] = ()
    children: tuple[tuple[str, "Shape"], ...] = ()
    variants: tuple[tuple[str, dict[str, "Shape"]], ...] = ()
    known: frozenset[str] = field(default_factory=frozenset)


def assemble(
    headers: list[tuple[str, list[str]]], entries: list[Entry]
) -> list[dict[str, Any]]:
    """Return every sheet, in order, each with its header and rows.

    headers -- each sheet's name and header, in the order of the Sheets
    entries -- each row with the name of its sheet, in input order

    spec › file.4: every sheet of the Sheets, in their order, each with
    its header, and no records where the input holds none. Raise
    KeyError for a row of a sheet that is not in the headers.
    """
    rows: dict[str, list[list[str]]] = {name: [] for name, _ in headers}
    for name, row in entries:
        rows[name].append(row)
    return [
        {"sheet name": name, "header": header, "records": rows[name]}
        for name, header in headers
    ]


def shape_sheets(sheet: str, shape: Shape) -> list[tuple[str, list[str]]]:
    """Return the sheets of a definition, in order, with their headers.

    sheet -- the definition's sheet name
    shape -- the definition

    spec › file.4: each sheet is followed by its lines sheets, then by
    its child sheets.
    """
    sheets = [(sheet, ["address", "pointer", *shape.columns])]
    for name in shape.columns:
        if name in shape.lines:
            sheets.append((f"{sheet}.{name}", LINES))
    for name in shape.nodes:
        sheets += node_sheets(f"{sheet}.{name}")
    for name, child in shape.children:
        sheets += shape_sheets(f"{sheet}.{name}", child)
    sheets += node_sheets(f"{sheet}.additionalProperties")
    for name, variants in shape.variants:
        sheets += variant_sheets(f"{sheet}.{name}", variants)
    return sheets


def variant_sheets(
    sheet: str, variants: dict[str, Shape]
) -> list[tuple[str, list[str]]]:
    """Return the sheets of the definitions of an array, in order.

    sheet -- the sheet name the definitions' names extend
    variants -- each definition by the type value its data carries
    """
    sheets: list[tuple[str, list[str]]] = []
    for name, shape in variants.items():
        sheets += shape_sheets(f"{sheet}.{name}", shape)
    return sheets


def node_sheets(sheet: str) -> list[tuple[str, list[str]]]:
    """Return a node sheet and its lines sheet, with their headers.

    sheet -- the node sheet's name

    spec › line.1: a node sheet's value has a lines sheet, named with
    value.
    """
    return [(sheet, NODE), (f"{sheet}.value", LINES)]


def array_entries(
    sheet: str, shape: Shape, address: str, at: str, value: Any
) -> list[Entry]:
    """Return the rows of an array whose items share one definition.

    sheet -- the definition's sheet name
    shape -- the definition
    address -- the address of the record the value belongs to
    at -- the pointer to the array
    value -- the array; anything else has no rows

    spec › nested.2: order is kept by the pointer's indexes.
    """
    if not isinstance(value, list):
        return []
    entries: list[Entry] = []
    for index, item in enumerate(value):
        entries += item_entries(sheet, shape, address, pointer(at, index), item)
    return entries


def variant_entries(
    sheet: str, variants: dict[str, Shape], address: str, at: str, value: Any
) -> list[Entry]:
    """Return the rows of an array whose items name their definition.

    sheet -- the sheet name the definitions' names extend
    variants -- each definition by the type value its data carries
    address -- the address of the record the value belongs to
    at -- the pointer to the array
    value -- the array; anything else has no rows

    spec › item.1: a part or tool belongs to the definition its type
    value names; one whose type names no definition is generic.
    """
    if not isinstance(value, list):
        return []
    entries: list[Entry] = []
    for index, item in enumerate(value):
        name = item.get("type") if isinstance(item, dict) else None
        if not isinstance(name, str) or name not in variants:
            name = "generic"
        entries += item_entries(
            f"{sheet}.{name}", variants[name], address, pointer(at, index), item
        )
    return entries


def item_entries(
    sheet: str, shape: Shape, address: str, at: str, value: Any
) -> list[Entry]:
    """Return the rows of an object of a definition, and its children.

    sheet -- the definition's sheet name
    shape -- the definition
    address -- the address of the record the value belongs to
    at -- the pointer to the object
    value -- the object; anything else has no rows

    spec › nested.1: a nonsimple field is a child sheet that carries
    its parent's key. spec › text.1: an optional field defaults to
    null, and null is written as an absent field.
    """
    if not isinstance(value, dict):
        return []
    row = [address, cell(at)]
    row += [cell(text(value.get(name))) for name in shape.columns]
    entries: list[Entry] = [(sheet, row)]
    for name in shape.columns:
        if name in shape.lines:
            entries += text_entries(
                f"{sheet}.{name}", [address, cell(pointer(at, name))], value.get(name)
            )
    for name in shape.nodes:
        if value.get(name) is not None:
            entries += node_entries(
                f"{sheet}.{name}", [address], value[name], pointer(at, name)
            )
    for name, child in shape.children:
        entries += item_entries(
            f"{sheet}.{name}", child, address, pointer(at, name), value.get(name)
        )
    for name, member in value.items():
        if name not in shape.known:
            entries += node_entries(
                f"{sheet}.additionalProperties", [address], member, pointer(at, name)
            )
    for name, variants in shape.variants:
        entries += variant_entries(
            f"{sheet}.{name}", variants, address, pointer(at, name), value.get(name)
        )
    return entries


def node_entries(sheet: str, key: list[str], value: Any, at: str) -> list[Entry]:
    """Return the rows of a value of any shape, a row per node.

    sheet -- the node sheet's name
    key -- the columns before the pointer
    value -- the value
    at -- the pointer to the value

    spec › node.1: its pointer, its type and its value; a node before
    the nodes within it, members and elements as written.
    """
    entries: list[Entry] = [
        (sheet, key + [cell(at), kind(value), cell(text(value))])
    ]
    entries += text_entries(f"{sheet}.value", key + [cell(at)], value)
    if isinstance(value, dict):
        for name, member in value.items():
            entries += node_entries(sheet, key, member, pointer(at, name))
    elif isinstance(value, list):
        for index, element in enumerate(value):
            entries += node_entries(sheet, key, element, pointer(at, index))
    return entries


def text_entries(sheet: str, key: list[str], value: Any) -> list[Entry]:
    """Return the lines of a text value, a row per line.

    sheet -- the lines sheet's name
    key -- the columns before the line: the key of the text value
    value -- the value; a text without a line break has no lines

    spec › line.1: each line is a record of its lines sheet, keyed by
    the text value and line, the line's zero-based position.
    """
    return [
        (sheet, key + [str(index), cell(line)])
        for index, line in enumerate(lines(text(value)))
    ]


def decode(document: str) -> Any:
    """Read a JSON text, keeping each number as the text it writes.

    document -- the JSON text

    RFC 8259, Section 6: an implementation may limit the range and
    precision of numbers; spec › node.3. Raise ValueError for a text
    that is not JSON.
    """
    return json.loads(document, parse_int=Number, parse_float=Number)


def kind(value: Any) -> str:
    """Return the type of a JSON value, as spec › node.1 names it.

    value -- a decoded JSON value

    RFC 8259, Section 3: a value is false, null, true, an object, an
    array, a number or a string.
    """
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Number):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "null"


def text(value: Any) -> str | None:
    """Return the text of a string, or of a number as written.

    value -- a decoded JSON value

    Return None for any other value: spec › node.1, an object, an
    array, true, false and null have an empty value.
    """
    return value if isinstance(value, str) else None


def cell(value: str | None) -> str:
    """Return a text as a field is written: itself, or empty.

    value -- a text, or None

    spec › character.1: a field that holds HT, LF, FF or CR cannot be
    represented, and here it is left empty. spec › text.1: null is
    written as an empty field.
    """
    if value is None or not _SEPARATORS.isdisjoint(value):
        return ""
    return value


def lines(value: str | None) -> list[str]:
    """Return the lines of a text that holds a line break.

    value -- a text, or None

    Return no lines for a text without a line break. spec › line.1: a
    line break is LF, or CRLF; a line is the text between line breaks,
    so the lines joined with LF give the text back.
    """
    if value is None or "\n" not in value:
        return []
    found = value.split("\n")
    return [line.removesuffix("\r") for line in found[:-1]] + found[-1:]


def pointer(at: str, token: str | int) -> str:
    """Extend a JSON Pointer by one reference token.

    at -- the pointer to the parent value
    token -- an object member's name, or an array element's index

    RFC 6901, Section 3: each reference token is prefixed by '/'; in a
    token '~' is written '~0' and '/' is written '~1'.
    """
    name = str(token).replace("~", "~0").replace("/", "~1")
    return f"{at}/{name}"
