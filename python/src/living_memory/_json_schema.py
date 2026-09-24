"""Whether a JSON value validates against a JSON Schema of draft-07."""

__all__ = ["validates"]

from decimal import Decimal
from typing import Any

from living_memory import _json

# JSON-SCHEMA-07 draft-07-schema.json, written out by hand without its
# title annotation.
_METASCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "http://json-schema.org/draft-07/schema#",
    "definitions": {
        "schemaArray": {"type": "array", "minItems": 1, "items": {"$ref": "#"}},
        "nonNegativeInteger": {"type": "integer", "minimum": 0},
        "nonNegativeIntegerDefault0": {
            "allOf": [{"$ref": "#/definitions/nonNegativeInteger"}, {"default": 0}]
        },
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
        "maxLength": {"$ref": "#/definitions/nonNegativeInteger"},
        "minLength": {"$ref": "#/definitions/nonNegativeIntegerDefault0"},
        "pattern": {"type": "string", "format": "regex"},
        "additionalItems": {"$ref": "#"},
        "items": {
            "anyOf": [{"$ref": "#"}, {"$ref": "#/definitions/schemaArray"}],
            "default": True,
        },
        "maxItems": {"$ref": "#/definitions/nonNegativeInteger"},
        "minItems": {"$ref": "#/definitions/nonNegativeIntegerDefault0"},
        "uniqueItems": {"type": "boolean", "default": False},
        "contains": {"$ref": "#"},
        "maxProperties": {"$ref": "#/definitions/nonNegativeInteger"},
        "minProperties": {"$ref": "#/definitions/nonNegativeIntegerDefault0"},
        "required": {"$ref": "#/definitions/stringArray"},
        "additionalProperties": {"$ref": "#"},
        "definitions": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "default": {},
        },
        "properties": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "default": {},
        },
        "patternProperties": {
            "type": "object",
            "additionalProperties": {"$ref": "#"},
            "propertyNames": {"format": "regex"},
            "default": {},
        },
        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [{"$ref": "#"}, {"$ref": "#/definitions/stringArray"}]
            },
        },
        "propertyNames": {"$ref": "#"},
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
        "if": {"$ref": "#"},
        "then": {"$ref": "#"},
        "else": {"$ref": "#"},
        "allOf": {"$ref": "#/definitions/schemaArray"},
        "anyOf": {"$ref": "#/definitions/schemaArray"},
        "oneOf": {"$ref": "#/definitions/schemaArray"},
        "not": {"$ref": "#"},
    },
    "default": True,
}

# JSON-SCHEMA-07, 8.2.
_DOCUMENTS = {_METASCHEMA["$id"].removesuffix("#"): _METASCHEMA}


def validates(instance: Any, schema: Any, root: Any) -> bool:
    # JSON-SCHEMA-07, 4.3.1, 8.3 and 6.4: a keyword not supported is
    # ignored.
    if schema is True or schema is False:
        return schema
    if "$ref" in schema:
        target, document = _resolve(schema["$ref"], root)
        return validates(instance, target, document)
    return all(
        keyword(instance, schema, root)
        for name, keyword in _KEYWORDS.items()
        if name in schema
    )


def _equal(one: Any, other: Any) -> bool:
    # JSON-SCHEMA-07, 4.2.3.
    found = _json.type(one)
    if found != _json.type(other):
        return False
    if found == "number":
        return _decimal(one) == _decimal(other)
    if found == "array":
        return len(one) == len(other) and all(map(_equal, one, other))
    if found == "object":
        return one.keys() == other.keys() and all(
            _equal(one[name], other[name]) for name in one
        )
    return one == other


def _schema_type(instance: Any) -> str:
    # JSON-SCHEMA-07 validation, 6.1.1.
    found = _json.type(instance)
    if found == "number" and _decimal(instance) == _decimal(instance).to_integral():
        return "integer"
    return found


def _resolve(reference: str, root: Any) -> tuple[Any, Any]:
    # JSON-SCHEMA-07, 8.2 and 8.3; RFC 6901, Section 4.
    base, _, fragment = reference.partition("#")
    if base == "":
        document = root
    elif base in _DOCUMENTS:
        document = _DOCUMENTS[base]
    else:
        raise ValueError(f"no schema is held for {reference}")
    target = document
    for token in fragment.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        target = target[int(token)] if isinstance(target, list) else target[token]
    return target, document


def _type(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.1.1.
    value = schema["type"]
    names = value if isinstance(value, list) else [value]
    found = _schema_type(instance)
    return found in names or (found == "integer" and "number" in names)


def _enum(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.1.2.
    return any(_equal(instance, element) for element in schema["enum"])


def _const(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.1.3.
    return _equal(instance, schema["const"])


def _minimum(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.2.4.
    if _json.type(instance) != "number":
        return True
    return _decimal(instance) >= _decimal(schema["minimum"])


def _exclusive_minimum(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.2.5.
    if _json.type(instance) != "number":
        return True
    return _decimal(instance) > _decimal(schema["exclusiveMinimum"])


def _items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.4.1.
    if not isinstance(instance, list):
        return True
    return all(validates(element, schema["items"], root) for element in instance)


def _min_items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.4.4.
    if not isinstance(instance, list):
        return True
    return len(instance) >= _decimal(schema["minItems"])


def _unique_items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.4.5.
    if schema["uniqueItems"] is not True or not isinstance(instance, list):
        return True
    return not any(
        _equal(instance[i], instance[j])
        for i in range(len(instance))
        for j in range(i + 1, len(instance))
    )


def _required(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.5.3.
    if not isinstance(instance, dict):
        return True
    return all(name in instance for name in schema["required"])


def _properties(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.5.4.
    if not isinstance(instance, dict):
        return True
    return all(
        validates(instance[name], child, root)
        for name, child in schema["properties"].items()
        if name in instance
    )


def _additional_properties(
    instance: Any, schema: dict[str, Any], root: Any
) -> bool:
    # JSON-SCHEMA-07 validation, 6.5.6; no schema held applies
    # patternProperties.
    if not isinstance(instance, dict):
        return True
    named = schema.get("properties", {})
    return all(
        validates(value, schema["additionalProperties"], root)
        for name, value in instance.items()
        if name not in named
    )


def _property_names(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.5.8.
    if not isinstance(instance, dict):
        return True
    return all(validates(name, schema["propertyNames"], root) for name in instance)


def _all_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.7.1.
    return all(validates(instance, child, root) for child in schema["allOf"])


def _any_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # JSON-SCHEMA-07 validation, 6.7.2.
    return any(validates(instance, child, root) for child in schema["anyOf"])


_KEYWORDS = {
    "type": _type,
    "enum": _enum,
    "const": _const,
    "minimum": _minimum,
    "exclusiveMinimum": _exclusive_minimum,
    "items": _items,
    "minItems": _min_items,
    "uniqueItems": _unique_items,
    "required": _required,
    "properties": _properties,
    "additionalProperties": _additional_properties,
    "propertyNames": _property_names,
    "allOf": _all_of,
    "anyOf": _any_of,
}


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))
