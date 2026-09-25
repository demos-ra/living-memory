"""How a schema and its instances become related sheets."""

from __future__ import annotations

__all__ = ["Layout", "layout", "records", "sheets"]

from dataclasses import dataclass, replace
from typing import Any

from living_memory import _fields, _json, _json_schema
from living_memory._json_pointer import PlacedError, pointer

# A column holds one of these primitive types (spec › field.1).
_SIMPLE = frozenset({"string", "number", "boolean", "null"})
# These keywords apply subschemas to the same location, in the order
# JSON Schema Validation defines them (spec › file.3, sheet.3).
_SAME_LOCATION = ("dependencies", "if", "then", "else", "allOf", "anyOf", "oneOf")
# A string's sheet has these columns after its key (spec › field.3).
_TEXT = ("page", "line", "position", "value")

# A record is its sheet's index and its fields; a sheet is its name and
# its header; a write gives records and the pointers not carried.
_Record = tuple[int, list[str]]
_Sheet = tuple[str, list[str]]
_Written = tuple[list[_Record], list[str]]


@dataclass(frozen=True)
class Layout:
    # A layout holds the sheets a schema gives, in file order, and the
    # node that writes the input values (spec › file.2, file.3).
    sheets: list[_Sheet]
    root: Any


def layout(schema: Any) -> Layout:
    # The input values are one relation, a sheet named by the root
    # schema's title (spec › sheet.1, module.1).
    if _json.type(schema) != "object" or _json.type(schema.get("title")) != "string":
        raise PlacedError("the root schema holds no title", "")
    title = _name(schema["title"], "/title")
    node, found = _root(schema, _Where(schema, title, ("pointer",), 0, ""))
    return Layout(found, node)


def records(sheet_layout: Layout, value: Any, position: int) -> _Written:
    # An input value gives its records, each with its sheet's index, and
    # the pointers of the text it does not carry (spec › record.1-3,
    # field.4).
    return sheet_layout.root.write(value, pointer("", position), "")


def sheets(sheet_layout: Layout, found: list[_Record]) -> list[dict[str, Any]]:
    # Every sheet of the layout is given with its header and its
    # records, a sheet with no records included (spec › file.2).
    held: list[list[list[str]]] = [[] for _ in sheet_layout.sheets]
    for index, fields in found:
        held[index].append(fields)
    return [
        {"sheet name": name, "header": header, "records": sheet_records}
        for (name, header), sheet_records in zip(sheet_layout.sheets, held)
    ]


@dataclass(frozen=True)
class _Where:
    # A builder adds its sheet knowing the root schema, which every $ref
    # resolves against, the sheet's name, its key columns, its index in
    # file order, and the pointer to its schema, which places a name a
    # field cannot hold (spec › record.3, file.3, module.1).
    root: Any
    name: str
    keys: tuple[str, ...]
    first: int
    schema_at: str

    def below(self, label: str, first: int, schema_at: str) -> _Where:
        # A subordinate sheet is named by its parent's name, '.', and
        # its label, and begins with its parent's key, copied down
        # (spec › sheet.2, record.3).
        keys = (f"{self.name}.pointer", "pointer")
        return _Where(self.root, f"{self.name}.{label}", keys, first, schema_at)


def _root(schema: Any, where: _Where) -> tuple[Any, list[_Sheet]]:
    # The input values are an object's or an array's sheet, or else a
    # sheet of instances (spec › sheet.1, sheet.4).
    allowed = _allowed(schema, where.root)
    if allowed == {"object"}:
        return _object(schema, where)
    if allowed == {"array"}:
        return _array(schema, where)
    return _instances(where)


def _child(schema: Any, where: _Where) -> tuple[Any, list[_Sheet]]:
    # A child location's instances are the records of its sheet; one
    # simple type has the one column value (spec › sheet.2, sheet.4,
    # field.1).
    allowed = _allowed(schema, where.root)
    if allowed == {"object"}:
        return _object(schema, where)
    if allowed == {"array"}:
        return _array(schema, where)
    if len(allowed) == 1 and allowed <= _SIMPLE:
        below = where.below("value", where.first + 1, where.schema_at)
        field, added = _column(allowed, below)
        own = (where.name, [*where.keys, "value"])
        return _Value(where.first, field), [own, *added]
    return _instances(where)


def _property(schema: Any, where: _Where) -> tuple[Any, list[_Sheet]]:
    # A property that is not a column is a subordinate sheet where it
    # allows one type, object or array, an array whose items is one
    # schema giving the sheet of its elements; any other is a sheet of
    # instances (spec › sheet.2, sheet.4).
    allowed = _allowed(schema, where.root)
    if allowed == {"object"}:
        return _object(schema, where)
    if allowed == {"array"}:
        resolved, at = _resolve(schema, where.root, where.schema_at)
        items = resolved.get("items", True)
        if isinstance(items, list):
            return _array(resolved, replace(where, schema_at=at))
        node, added = _child(items, replace(where, schema_at=pointer(at, "items")))
        return _Elements(node), added
    return _instances(where)


def _object(
    schema: Any, where: _Where, excluded: frozenset[str] = frozenset()
) -> tuple[_Object, list[_Sheet]]:
    # Each object is one record, its simple required properties its
    # columns; its other members go to subordinate sheets, and the
    # subschemas applied to it give branch sheets (spec › sheet.2-5,
    # field.1).
    schema, at = _resolve(schema, where.root, where.schema_at)
    required = schema.get("required", [])
    columns: list[tuple[str, Any]] = []
    properties: dict[str, Any] = {}
    found: list[_Sheet] = []
    for prop, child in schema.get("properties", {}).items():
        if prop in excluded:
            continue
        prop_at = pointer(pointer(at, "properties"), prop)
        below = where.below(_name(prop, prop_at), where.first + 1 + len(found), prop_at)
        allowed = _allowed(child, where.root)
        if prop in required and len(allowed) == 1 and allowed <= _SIMPLE:
            field, added = _column(allowed, below)
            columns.append((prop, field))
        else:
            properties[prop], added = _property(child, below)
        found += added
    patterns: list[tuple[str, Any]] = []
    pattern_schemas = schema.get("patternProperties", {})
    for pattern, child in pattern_schemas.items():
        pattern_at = pointer(pointer(at, "patternProperties"), pattern)
        label = "patternProperties"
        if len(pattern_schemas) > 1:
            label += f".{_name(pattern, pattern_at)}"
        below = where.below(label, where.first + 1 + len(found), pattern_at)
        node, added = _child(child, below)
        patterns.append((pattern, node))
        found += added
    additional: tuple[Any, ...] = ()
    rest = schema.get("additionalProperties", True)
    if rest is not False:
        rest_at = pointer(at, "additionalProperties")
        below = where.below(
            "additionalProperties", where.first + 1 + len(found), rest_at
        )
        node, added = _child(rest, below)
        additional = (node,)
        found += added
    owned = excluded | frozenset(schema.get("properties", {}))
    branches: list[_Branch] = []
    for keyword in _SAME_LOCATION:
        for label, child, condition, child_at in _same_location(schema, keyword, at):
            if _allowed(child, where.root) != {"object"}:
                continue
            resolved, resolved_at = _resolve(child, where.root, child_at)
            if "title" in resolved:
                title = _name(resolved["title"], pointer(resolved_at, "title"))
            else:
                title = _name(label, child_at)
            first = where.first + 1 + len(found)
            below = where.below(f"{keyword}.{title}", first, resolved_at)
            node, added = _object(resolved, below, owned)
            branches.append(_Branch(keyword, condition, node, where.root))
            found += added
    node = _Object(
        where.first,
        len(where.keys) > 1,
        schema,
        tuple(columns),
        properties,
        tuple(patterns),
        additional,
        tuple(branches),
    )
    own = (where.name, [*where.keys, *(name for name, _ in columns)])
    return node, [own, *found]


def _same_location(
    schema: dict[str, Any], keyword: str, at: str
) -> list[tuple[str, Any, Any, str]]:
    # Each subschema a keyword applies to the same location comes with
    # its key or zero-based position, with what decides whether it
    # collects the instance, a dependency's key, the if of then and
    # else, else itself, and with its pointer (spec › sheet.3).
    keyword_at = pointer(at, keyword)
    if keyword == "dependencies":
        return [
            (key, dependency, key, pointer(keyword_at, key))
            for key, dependency in schema.get("dependencies", {}).items()
            if not isinstance(dependency, list)
        ]
    if keyword in ("if", "then", "else"):
        if keyword in schema and "if" in schema:
            return [("0", schema[keyword], schema["if"], keyword_at)]
        return []
    return [
        (str(i), child, child, pointer(keyword_at, i))
        for i, child in enumerate(schema.get(keyword, []))
    ]


def _array(schema: Any, where: _Where) -> tuple[_Array, list[_Sheet]]:
    # Each array is one record; where items is one schema, its sheet
    # holds every element, and else each schema of items and
    # additionalItems has a sheet holding the elements at its positions
    # (spec › sheet.2).
    schema, at = _resolve(schema, where.root, where.schema_at)
    keyed = len(where.keys) > 1
    own = (where.name, list(where.keys))
    items = schema.get("items", True)
    items_at = pointer(at, "items")
    if not isinstance(items, list):
        below = where.below("items", where.first + 1, items_at)
        node, added = _child(items, below)
        return _Array(where.first, keyed, (), (node,)), [own, *added]
    nodes = []
    found: list[_Sheet] = []
    for i, item in enumerate(items):
        item_at = pointer(items_at, i)
        below = where.below(f"items.{i}", where.first + 1 + len(found), item_at)
        node, added = _child(item, below)
        nodes.append(node)
        found += added
    additional: tuple[Any, ...] = ()
    rest = schema.get("additionalItems", True)
    if rest is not False:
        rest_at = pointer(at, "additionalItems")
        below = where.below("additionalItems", where.first + 1 + len(found), rest_at)
        node, added = _child(rest, below)
        additional = (node,)
        found += added
    return _Array(where.first, keyed, tuple(nodes), additional), [own, *found]


def _instances(where: _Where) -> tuple[_Instances, list[_Sheet]]:
    # A sheet of instances holds its keys, the type and the value
    # (spec › sheet.4, field.2).
    text, added = _text(where.below("value", where.first + 1, where.schema_at))
    own = (where.name, [*where.keys, "type", "value"])
    return _Instances(where.first, len(where.keys) > 1, text), [own, *added]


def _column(allowed: set[str], where: _Where) -> tuple[Any, list[_Sheet]]:
    # A string column has its string's sheet; a number, a boolean or
    # null is its text (spec › field.2, field.3).
    if allowed == {"string"}:
        return _text(where)
    return _Plain(), []


def _text(where: _Where) -> tuple[_Text, list[_Sheet]]:
    # A string's sheet is named by its sheet's name, '.', and its
    # column's name, and begins with the pointer of the record that
    # holds the string (spec › field.3, record.3).
    return _Text(where.first), [(where.name, [where.keys[0], *_TEXT])]


def _allowed(schema: Any, root: Any) -> set[str]:
    # The types a location allows are its type keyword's, all six
    # without one, narrowed by allOf and by the union of anyOf and
    # oneOf; an integer is a number (spec › sheet.1, value.2).
    schema = _resolve(schema, root, "")[0]
    if schema is False:
        return set()
    if "type" in schema:
        listed = schema["type"]
        names = listed if isinstance(listed, list) else [listed]
        types = {"number" if name == "integer" else name for name in names}
    else:
        types = set(_json.TYPES)
    for child in schema.get("allOf", []):
        types &= _allowed(child, root)
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            types &= set().union(*(_allowed(c, root) for c in schema[keyword]))
    return types


def _resolve(schema: Any, root: Any, at: str) -> tuple[Any, str]:
    # A $ref is read as the schema it references, at that schema's
    # pointer, and true as the empty schema (spec › module.3).
    while isinstance(schema, dict) and "$ref" in schema:
        schema, at = _json_schema.resolve(schema["$ref"], root)
    return ({} if schema is True else schema), at


@dataclass(frozen=True)
class _Plain:
    # A number, a boolean or null is written as its text (spec ›
    # field.2).
    def field(self, value: Any, at: str, owner: str) -> tuple[str, _Written]:
        return _fields.text(value), ([], [])


@dataclass(frozen=True)
class _Text:
    # A string is written with what a field cannot hold left out and
    # reported; one holding FF, a line break or HT is an empty field,
    # and its runs are the records of its string's sheet (spec ›
    # field.2-4).
    index: int

    def field(self, value: Any, at: str, owner: str) -> tuple[str, _Written]:
        if _json.type(value) != "string":
            return _fields.text(value), ([], [])
        kept, dropped = _fields.carried(value)
        reports = [at] if dropped else []
        if not _fields.holds_separator(kept):
            return kept, ([], reports)
        runs = [
            (self.index, [owner, str(page), str(line), str(position), run])
            for page, line, position, run in _fields.runs(kept)
        ]
        return "", (runs, reports)


@dataclass(frozen=True)
class _Value:
    # A child instance of one simple type is written in the column
    # value (spec › field.1).
    index: int
    field: Any

    def write(self, value: Any, at: str, parent: str) -> _Written:
        text, (found, reports) = self.field.field(value, at, at)
        return [(self.index, [parent, at, text]), *found], reports


@dataclass(frozen=True)
class _Instances:
    # Each instance within the value is one record, the value itself
    # first (spec › sheet.4, field.2, record.2).
    index: int
    keyed: bool
    text: _Text

    def write(self, value: Any, at: str, parent: str) -> _Written:
        keys = [parent, at] if self.keyed else [at]
        text, (found, reports) = self.text.field(value, at, at)
        found = [(self.index, [*keys, _json.type(value), text]), *found]
        for token, child in _children(value):
            where, missed = _member(at, token)
            child_records, child_reports = self.write(child, where, parent)
            found += child_records
            reports += missed + child_reports
        return found, reports


@dataclass(frozen=True)
class _Array:
    # Each array is one record; each element goes to the sheet of the
    # items schema at its position, else to additionalItems' (spec ›
    # sheet.2).
    index: int
    keyed: bool
    items: tuple[Any, ...]
    additional: tuple[Any, ...]

    def write(self, value: Any, at: str, parent: str) -> _Written:
        found: list[_Record] = [(self.index, [parent, at] if self.keyed else [at])]
        reports: list[str] = []
        for position, element in enumerate(value):
            for node in self.items[position : position + 1] or self.additional:
                element_records, element_reports = node.write(
                    element, pointer(at, position), at
                )
                found += element_records
                reports += element_reports
        return found, reports


@dataclass(frozen=True)
class _Elements:
    # A property's array gives its elements as the records of the sheet
    # named by the property, keyed by the record that holds the array
    # (spec › sheet.2, record.3).
    sheet: Any

    def write(self, value: Any, at: str, parent: str) -> _Written:
        found: list[_Record] = []
        reports: list[str] = []
        for position, element in enumerate(value):
            element_records, element_reports = self.sheet.write(
                element, pointer(at, position), parent
            )
            found += element_records
            reports += element_reports
        return found, reports


@dataclass(frozen=True)
class _Branch:
    # A subschema applied to the same location collects the instance's
    # annotations under a dependency's key, where then's if is valid,
    # where else's if is not, and else where the instance is valid
    # against it (JSON Schema Validation, 3.3.1. Annotations and
    # Validation Outcomes; spec › sheet.3).
    keyword: str
    condition: Any
    sheet: _Object
    root: Any

    def collects(self, instance: Any) -> bool:
        if self.keyword == "dependencies":
            return self.condition in instance
        valid = _json_schema.validates(instance, self.condition, self.root)
        return not valid if self.keyword == "else" else valid


@dataclass(frozen=True)
class _Object:
    # Each object is one record of its columns; its other members go to
    # subordinate sheets, and it goes to the branches that collect it
    # (spec › sheet.2-5).
    index: int
    keyed: bool
    schema: dict[str, Any]
    columns: tuple[tuple[str, Any], ...]
    properties: dict[str, Any]
    patterns: tuple[tuple[str, Any], ...]
    additional: tuple[Any, ...]
    branches: tuple[_Branch, ...]

    def write(self, value: Any, at: str, parent: str) -> _Written:
        return self.held(value, at, parent, frozenset(value))

    def held(self, value: Any, at: str, parent: str, names: frozenset[str]) -> _Written:
        # The object's record holds the members placed on it; each
        # branch that collects the object writes the members placed on
        # it.
        collecting = [branch for branch in self.branches if branch.collects(value)]
        place = self._place(value, collecting, names)
        found: list[_Record] = []
        reports: list[str] = []
        fields = []
        for name, field in self.columns:
            text = ""
            if name in place and place[name] is self:
                text, (text_records, text_reports) = field.field(
                    value[name], pointer(at, name), at
                )
                found += text_records
                reports += text_reports
            fields.append(text)
        keys = [parent, at] if self.keyed else [at]
        found = [(self.index, keys + fields), *found]
        column_names = {name for name, _ in self.columns}
        for name, member in value.items():
            if name in column_names or name not in place or place[name] is not self:
                continue
            where, missed = _member(at, name)
            member_records, member_reports = self._node(name).write(member, where, at)
            found += member_records
            reports += missed + member_reports
        for branch in collecting:
            placed = frozenset(
                n for n, holder in place.items() if holder is branch.sheet
            )
            branch_records, branch_reports = branch.sheet.held(value, at, at, placed)
            found += branch_records
            reports += branch_reports
        return found, reports

    def names(self, name: str) -> bool:
        return _json_schema.named(name, self.schema)

    def _place(
        self, value: dict[str, Any], collecting: list[_Branch], names: frozenset[str]
    ) -> dict[str, _Object]:
        # Each member is written once: by this schema where it names or
        # matches it, else by the first branch that collects the value
        # and names it, else by the first additionalProperties that
        # applies (spec › sheet.5).
        place: dict[str, _Object] = {}
        for name in value:
            if name not in names:
                continue
            naming = [b.sheet for b in collecting if b.sheet.names(name)]
            taking = [b.sheet for b in collecting if b.sheet.additional]
            if self.names(name):
                place[name] = self
            elif naming:
                place[name] = naming[0]
            elif self.additional:
                place[name] = self
            elif taking:
                place[name] = taking[0]
        return place

    def _node(self, name: str) -> Any:
        # A member's node is its property's, else the first matching
        # pattern's, else additionalProperties'.
        if name in self.properties:
            return self.properties[name]
        for pattern, node in self.patterns:
            if _json_schema.named(name, {"patternProperties": {pattern: {}}}):
                return node
        return self.additional[0]


def _member(at: str, name: str | int) -> tuple[str, list[str]]:
    # A member name within a pointer holds no HT, LF, FF or CR; what is
    # left out is reported by the pointer as written (spec › field.4).
    if isinstance(name, int):
        return pointer(at, name), []
    kept, dropped = _fields.name_carried(name)
    return pointer(at, kept), [pointer(at, name)] if dropped else []


def _children(value: Any) -> list[tuple[str | int, Any]]:
    # An object's children are its members, and an array's its
    # elements, in order.
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, list):
        return list(enumerate(value))
    return []


def _name(text: Any, at: str) -> str:
    # A name the schema gives a sheet or a column is text a field can
    # hold; one that is not is placed by its pointer in the schema
    # (spec › module.1).
    if _json.type(text) != "string" or _fields.name_carried(text)[1]:
        raise PlacedError(f"{text!r} is not a name a field can hold", at)
    return text
