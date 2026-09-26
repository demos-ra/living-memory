"""What a module specification supplies, and how its schema is read."""

__all__ = ["read", "resolve", "taken"]

from collections.abc import Callable
from typing import Any

from living_memory import _json, _json_pointer, _json_schema, _separators
from living_memory._json_pointer import PlacedError

# The keywords that apply subschemas, to child locations and to the same
# location, in the order JSON Schema Validation defines them (spec ›
# module.2).
_TAKEN = (
    "items",
    "additionalItems",
    "properties",
    "patternProperties",
    "additionalProperties",
    "dependencies",
    "if",
    "then",
    "else",
    "allOf",
    "anyOf",
    "oneOf",
)
# These keywords hold a list of subschemas.
_LIST = ("allOf", "anyOf", "oneOf")
# These keywords apply a subschema to the same instance location (JSON
# Schema Validation, 6.5.7. dependencies; 6.6. Keywords for Applying
# Subschemas Conditionally; 6.7. Keywords for Applying Subschemas With
# Boolean Logic).
_SAME = ("dependencies", "if", "then", "else", "allOf", "anyOf", "oneOf", "not")


def read(schema: bytes) -> Any:
    # The schema is a JSON text whose root schema holds a title; every
    # name it gives a sheet or a column is text a field can hold, every
    # pattern is of the subset, and every $ref resolves within it to a
    # schema other than the root, and runs into no loop (spec ›
    # module.1, module.2).
    root = _json.decode(schema)
    if _json.type(root) != "object" or "title" not in root:
        raise PlacedError("the root schema holds no title", "")
    _check(root, root, "")
    _loops(root, root, "")
    return root


def resolve(schema: Any, root: Any, at: str) -> tuple[Any, str]:
    # A $ref is read as the schema it references, at that schema's
    # pointer, its other members ignored; true is the empty schema (JSON
    # Schema, 4.3.1. JSON Schema Values and Keywords; 8.3. Schema
    # References With "$ref"; spec › module.2).
    while isinstance(schema, dict) and "$ref" in schema:
        schema, at = _json_schema.resolve(schema["$ref"], root)
    return ({} if schema is True else schema), at


def taken(schema: dict[str, Any], at: str) -> list[tuple[str, Any, Any, str]]:
    # A schema's subschemas are taken from its keywords in the order
    # JSON Schema Validation defines them, and within a keyword in the
    # order the schema writes them, each with its keyword and its place
    # in the schema: the keyword's, then its key or position; an
    # omitted items, additionalItems or additionalProperties is the
    # empty schema, additionalItems only where items is a list; then
    # and else are ignored without if, and a dependency that lists
    # names applies no subschema (JSON Schema Validation, 6.4.1.
    # items; 6.4.2. additionalItems; 6.5.6. additionalProperties; 6.5.7.
    # dependencies; 6.6. Keywords for Applying Subschemas Conditionally;
    # spec › module.2).
    found: list[tuple[str, Any, Any]] = []
    items = schema.get("items", True)
    for keyword in _TAKEN:
        if keyword == "items":
            if isinstance(items, list):
                found += [(keyword, i, child) for i, child in enumerate(items)]
            else:
                found.append((keyword, None, items))
        elif keyword in ("properties", "patternProperties", "dependencies"):
            found += [
                (keyword, key, child)
                for key, child in schema.get(keyword, {}).items()
                if not isinstance(child, list)
            ]
        elif keyword in _LIST:
            found += [(keyword, i, c) for i, c in enumerate(schema.get(keyword, []))]
        elif keyword == "additionalItems" and not isinstance(items, list):
            continue
        elif keyword in ("then", "else") and "if" not in schema:
            continue
        elif keyword in schema:
            found.append((keyword, None, schema[keyword]))
        elif keyword in ("additionalItems", "additionalProperties"):
            found.append((keyword, None, True))
    return [
        (keyword, key, child, _at(at, keyword, key)) for keyword, key, child in found
    ]


def _at(at: str, keyword: str, key: str | int | None) -> str:
    # A subschema's place: its keyword's, then its key or position
    # where the keyword holds several.
    keyword_at = _json_pointer.pointer(at, keyword)
    return keyword_at if key is None else _json_pointer.pointer(keyword_at, key)


def _check(schema: Any, root: Any, at: str) -> None:
    # Each subschema is checked in turn, a failure placed by its pointer
    # in the schema.
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        _reference(schema["$ref"], root, _json_pointer.pointer(at, "$ref"))
    if "title" in schema:
        _name(schema["title"], _json_pointer.pointer(at, "title"))
    for keyword in ("properties", "patternProperties", "dependencies"):
        keyword_at = _json_pointer.pointer(at, keyword)
        for key in schema.get(keyword, {}):
            _name(key, _json_pointer.pointer(keyword_at, key))
    if _json.type(schema.get("pattern")) == "string":
        _placed(lambda: _json_schema.compile_pattern(schema["pattern"]), at, "pattern")
    patterns_at = _json_pointer.pointer(at, "patternProperties")
    for pattern in schema.get("patternProperties", {}):
        _placed(lambda: _json_schema.compile_pattern(pattern), patterns_at, pattern)
    for child_at, child in _json_schema.subschemas(schema, at):
        _check(child, root, child_at)


def _loops(schema: Any, root: Any, at: str) -> None:
    # Every $ref is followed through the subschemas applied to the same
    # instance location (spec › module.1).
    if isinstance(schema, dict) and "$ref" in schema:
        _loop(schema, root, (at,))
    for child_at, child in _json_schema.subschemas(schema, at):
        _loops(child, root, child_at)


def _loop(schema: Any, root: Any, path: tuple[str, ...]) -> None:
    # A schema must not be run into an infinite loop against a schema:
    # a $ref that reaches again, at the same instance location, a
    # schema it is applied from is placed by that $ref (JSON Schema,
    # 8.3. Schema References With "$ref"; spec › module.1).
    if not isinstance(schema, dict):
        return
    at = path[-1]
    if "$ref" in schema:
        target, target_at = _json_schema.resolve(schema["$ref"], root)
        if target_at in path:
            ref_at = _json_pointer.pointer(at, "$ref")
            raise PlacedError("a $ref runs its schema into a loop", ref_at)
        _loop(target, root, (*path, target_at))
        return
    depth = len(_json_pointer.tokens(at))
    for child_at, child in _json_schema.subschemas(schema, at):
        if _json_pointer.tokens(child_at)[depth] in _SAME:
            _loop(child, root, (*path, child_at))


def _reference(reference: str, root: Any, at: str) -> None:
    # A $ref resolves within the schema to a schema other than the
    # root, which is the sheet of the input values and no kind; its
    # last reference token may name the kind's sheet (spec › module.1,
    # module.2, relation.1, sheet.1).
    try:
        _, ref_at = _json_schema.resolve(reference, root)
    except ValueError as error:
        raise PlacedError(str(error), at) from None
    if ref_at == "":
        raise PlacedError("a $ref references the root schema", at)
    _name(_json_pointer.tokens(ref_at)[-1], at)


def _name(text: Any, at: str) -> None:
    # A title, a property's name, a key of patternProperties or
    # dependencies and the last reference token of a $ref are text a
    # field can hold (MTSV draft, Generators; spec › module.1).
    if _json.type(text) != "string" or _separators.cannot_hold(text):
        raise PlacedError(f"{text!r} is not a name a field can hold", at)


def _placed(checked: Callable[[], Any], at: str, token: str) -> None:
    # A check that fails is placed by the pointer to what it checked.
    try:
        checked()
    except ValueError as error:
        raise PlacedError(str(error), _json_pointer.pointer(at, token)) from None
