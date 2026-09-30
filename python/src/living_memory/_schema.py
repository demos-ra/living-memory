"""What a module specification supplies, and how its schema is read."""

__all__ = ["read", "resolve", "taken"]

from collections.abc import Callable
from typing import Any

from living_memory import _json, _json_pointer, _json_schema, _separators
from living_memory._json_pointer import PlacedError

# The keywords that apply subschemas, to child locations and to the same
# location, in the order JSON Schema Validation presents them (spec ›
# schema.11).
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
# The draft-07 metaschema, written out from draft-07-schema.json: what a
# schema of draft-07 is (JSON Schema; spec › schema.1).
_REFERENCE = {"$ref": "#"}
_NON_NEGATIVE = {"$ref": "#/definitions/nonNegativeInteger"}
_NON_NEGATIVE_0 = {"$ref": "#/definitions/nonNegativeIntegerDefault0"}
_SCHEMA_ARRAY = {"$ref": "#/definitions/schemaArray"}
_SCHEMA_OBJECT = {"type": "object", "additionalProperties": _REFERENCE, "default": {}}
_METASCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "http://json-schema.org/draft-07/schema#",
    "title": "Core schema meta-schema",
    "definitions": {
        "schemaArray": {"type": "array", "minItems": 1, "items": _REFERENCE},
        "nonNegativeInteger": {"type": "integer", "minimum": 0},
        "nonNegativeIntegerDefault0": {"allOf": [_NON_NEGATIVE, {"default": 0}]},
        "simpleTypes": {
            "enum": [
                "array",
                "boolean",
                "integer",
                "null",
                "number",
                "object",
                "string",
            ]
        },
        "stringArray": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
            "default": [],
        },
    },
    "type": ["object", "boolean"],
    "properties": {
        "$id": {"type": "string", "format": "uri-reference"},
        "$schema": {"type": "string", "format": "uri"},
        "$ref": {"type": "string", "format": "uri-reference"},
        "$comment": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "default": True,
        "readOnly": {"type": "boolean", "default": False},
        "writeOnly": {"type": "boolean", "default": False},
        "examples": {"type": "array", "items": True},
        "multipleOf": {"type": "number", "exclusiveMinimum": 0},
        "maximum": {"type": "number"},
        "exclusiveMaximum": {"type": "number"},
        "minimum": {"type": "number"},
        "exclusiveMinimum": {"type": "number"},
        "maxLength": _NON_NEGATIVE,
        "minLength": _NON_NEGATIVE_0,
        "pattern": {"type": "string", "format": "regex"},
        "additionalItems": _REFERENCE,
        "items": {"anyOf": [_REFERENCE, _SCHEMA_ARRAY], "default": True},
        "maxItems": _NON_NEGATIVE,
        "minItems": _NON_NEGATIVE_0,
        "uniqueItems": {"type": "boolean", "default": False},
        "contains": _REFERENCE,
        "maxProperties": _NON_NEGATIVE,
        "minProperties": _NON_NEGATIVE_0,
        "required": {"$ref": "#/definitions/stringArray"},
        "additionalProperties": _REFERENCE,
        "definitions": _SCHEMA_OBJECT,
        "properties": _SCHEMA_OBJECT,
        "patternProperties": {**_SCHEMA_OBJECT, "propertyNames": {"format": "regex"}},
        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [_REFERENCE, {"$ref": "#/definitions/stringArray"}]
            },
        },
        "propertyNames": _REFERENCE,
        "const": True,
        "enum": {"type": "array", "items": True, "minItems": 1, "uniqueItems": True},
        "type": {
            "anyOf": [
                {"$ref": "#/definitions/simpleTypes"},
                {
                    "type": "array",
                    "items": {"$ref": "#/definitions/simpleTypes"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
            ]
        },
        "format": {"type": "string"},
        "contentMediaType": {"type": "string"},
        "contentEncoding": {"type": "string"},
        "if": _REFERENCE,
        "then": _REFERENCE,
        "else": _REFERENCE,
        "allOf": _SCHEMA_ARRAY,
        "anyOf": _SCHEMA_ARRAY,
        "oneOf": _SCHEMA_ARRAY,
        "not": _REFERENCE,
    },
    "default": True,
}


def read(schema: bytes) -> Any:
    # The schema is a JSON text and a schema of draft-07, whose root
    # schema holds a title; every name it gives a sheet or a column is
    # text a field can hold, every pattern is of the subset, and every
    # $ref resolves within it by a JSON Pointer and runs into no loop
    # (spec › schema.1-8).
    root = _json.decode(schema)
    if not _json_schema.validates(root, _METASCHEMA, _METASCHEMA):
        place = _json_schema.locate(root, _METASCHEMA, _METASCHEMA)
        raise PlacedError("not a schema of draft-07", place)
    if _json.primitive_type(root) != "object" or "title" not in root:
        raise PlacedError("the root schema holds no title", "")
    _check(root)
    _loops(root)
    return root


def resolve(schema: Any, root: Any, at: str) -> tuple[Any, str]:
    # A $ref is read as the schema it references, at that schema's
    # pointer, its other members ignored; true is the empty schema (JSON
    # Schema, 4.3.1. JSON Schema Values and Keywords; 8.3. Schema
    # References With "$ref"; spec › schema.10).
    while isinstance(schema, dict) and "$ref" in schema:
        schema, at = _json_schema.resolve(schema["$ref"], root)
    return ({} if schema is True else schema), at


def taken(schema: dict[str, Any], at: str) -> list[tuple[str, Any, Any, str]]:
    # A schema's subschemas are taken from its keywords in the order
    # JSON Schema Validation presents them, and within a keyword in the
    # order the schema writes them, each with its keyword and its place
    # in the schema: the keyword's, then its key or position; an
    # omitted items, additionalItems or additionalProperties is the
    # empty schema, additionalItems only where items is a list; then
    # and else are ignored without if, and a dependency that lists
    # names applies no subschema (JSON Schema Validation, 6.4.1.
    # items; 6.4.2. additionalItems; 6.5.6. additionalProperties; 6.5.7.
    # dependencies; 6.6. Keywords for Applying Subschemas Conditionally;
    # spec › schema.11).
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


def _check(root: Any) -> None:
    # Each subschema is checked in turn, in the order the schema holds
    # them, a failure placed by its pointer in the schema; those still
    # to check are kept in a list, so a schema of any depth of nesting
    # is checked (spec › schema.1).
    waiting: list[tuple[str, Any]] = [("", root)]
    while waiting:
        at, schema = waiting.pop()
        if not isinstance(schema, dict):
            continue
        if "$ref" in schema:
            _reference(schema["$ref"], root, _json_pointer.pointer(at, "$ref"))
        if "title" in schema:
            _name(schema["title"], _json_pointer.pointer(at, "title"))
        for keyword in ("properties", "patternProperties", "dependencies"):
            keyword_at = _json_pointer.pointer(at, keyword)
            for key in schema.get(keyword, {}):
                _name(key, _json_pointer.pointer(keyword_at, key))
        if _json.primitive_type(schema.get("pattern")) == "string":
            pattern = schema["pattern"]
            _placed(lambda: _json_schema.compile_pattern(pattern), at, "pattern")
        patterns_at = _json_pointer.pointer(at, "patternProperties")
        for key in schema.get("patternProperties", {}):
            _placed(lambda: _json_schema.compile_pattern(key), patterns_at, key)
        waiting += reversed(_json_schema.subschemas(schema, at))


def _loops(root: Any) -> None:
    # Every $ref is followed through the subschemas applied to the same
    # instance location (spec › schema.7).
    waiting: list[tuple[str, Any]] = [("", root)]
    while waiting:
        at, schema = waiting.pop()
        if isinstance(schema, dict) and "$ref" in schema:
            _loop(schema, root, at)
        waiting += reversed(_json_schema.subschemas(schema, at))


def _loop(schema: Any, root: Any, at: str) -> None:
    # A schema must not be run into an infinite loop against a schema:
    # a $ref that reaches again, at the same instance location, a
    # schema it is applied from is placed by that $ref; each schema
    # still to follow is kept in a list with the schemas it is applied
    # from (JSON Schema, 8.3. Schema References With "$ref"; spec ›
    # schema.7).
    waiting: list[tuple[Any, tuple[str, ...]]] = [(schema, (at,))]
    while waiting:
        schema, path = waiting.pop()
        if not isinstance(schema, dict):
            continue
        at = path[-1]
        if "$ref" in schema:
            target, target_at = _json_schema.resolve(schema["$ref"], root)
            if target_at in path:
                ref_at = _json_pointer.pointer(at, "$ref")
                raise PlacedError("a $ref runs its schema into a loop", ref_at)
            waiting.append((target, (*path, target_at)))
            continue
        depth = len(_json_pointer.tokens(at))
        for child_at, child in reversed(_json_schema.subschemas(schema, at)):
            if _json_pointer.tokens(child_at)[depth] in _SAME:
                waiting.append((child, (*path, child_at)))


def _reference(reference: str, root: Any, at: str) -> None:
    # A $ref resolves within the schema, by a JSON Pointer; its last
    # reference token, where it has one, may name the kind's sheet
    # (spec › schema.3, schema.5, schema.6, sheet.2).
    try:
        _, ref_at = _json_schema.resolve(reference, root)
    except ValueError as error:
        raise PlacedError(str(error), at) from None
    if ref_at:
        _name(_json_pointer.tokens(ref_at)[-1], at)


def _name(text: Any, at: str) -> None:
    # A title, a property's name, a key of patternProperties or
    # dependencies and the last reference token of a $ref are text a
    # field can hold (MTSV draft, Generators; spec › schema.3).
    if _json.primitive_type(text) != "string" or _separators.cannot_hold(text):
        raise PlacedError(f"{text!r} is not a name a field can hold", at)


def _placed(checked: Callable[[], Any], at: str, token: str) -> None:
    # A check that fails is placed by the pointer to what it checked.
    try:
        checked()
    except ValueError as error:
        raise PlacedError(str(error), _json_pointer.pointer(at, token)) from None
