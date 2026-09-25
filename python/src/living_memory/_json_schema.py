"""Whether a JSON value validates against a JSON Schema of draft-07."""

__all__ = ["check", "locate", "named", "resolve", "validates"]

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from living_memory import _json, _json_pointer, _regular_expression

# These keywords hold a schema, a list of schemas, or an object of
# schemas (JSON Schema Validation, 3.1. Applicability).
_SCHEMA = ("additionalItems", "contains", "propertyNames", "not")
_SCHEMA += ("if", "then", "else", "additionalProperties")
_SCHEMAS = ("allOf", "anyOf", "oneOf")
_SCHEMA_OBJECTS = ("properties", "patternProperties", "definitions")


def validates(instance: Any, schema: Any, root: Any) -> bool:
    # A schema is true, false or an object; a $ref is used in its place,
    # its other members ignored; every assertion applies, and format and
    # the content keywords are not asserted (JSON Schema, 4.3.1. JSON
    # Schema Values and Keywords; 8.3. Schema References With "$ref";
    # spec › module.3, value.3).
    if schema is True or schema is False:
        return schema
    if "$ref" in schema:
        return validates(instance, resolve(schema["$ref"], root)[0], root)
    return all(
        keyword(instance, schema, root)
        for name, keyword in _KEYWORDS.items()
        if name in schema
    )


def locate(instance: Any, schema: Any, root: Any, at: str = "") -> str:
    # For an instance that does not validate, the place named is the
    # deepest instance location where an assertion fails: a child
    # location whose subschema fails, else a failing subschema allOf
    # applies here, else this location (JSON Schema Validation, 3.1.
    # Applicability; spec › value.3).
    if schema is False:
        return at
    if "$ref" in schema:
        return locate(instance, resolve(schema["$ref"], root)[0], root, at)
    for token, child, child_schema in _applied(instance, schema):
        if not validates(child, child_schema, root):
            return locate(child, child_schema, root, _json_pointer.pointer(at, token))
    for child_schema in schema.get("allOf", []):
        if not validates(instance, child_schema, root):
            return locate(instance, child_schema, root, at)
    return at


def resolve(reference: str, root: Any) -> tuple[Any, str]:
    # A reference resolves within the supplied schema, the one schema a
    # converter is given, to the schema and its pointer; any other is
    # non-conforming (JSON Schema, 8.3.1. Loading a referenced schema;
    # 8.3.2. Dereferencing; spec › module.3).
    base, _, fragment = reference.partition("#")
    if base:
        raise ValueError(f"$ref {reference!r} is outside the supplied schema")
    try:
        return _json_pointer.evaluate(root, fragment), fragment
    except (KeyError, IndexError, ValueError, TypeError):
        raise ValueError(f"$ref {reference!r} resolves to no schema") from None


def check(schema: Any, root: Any, at: str = "") -> None:
    # Every $ref resolves within the schema, and every pattern uses only
    # the tokens schema authors should limit themselves to; a failure is
    # placed by the pointer to the keyword or the pattern (spec ›
    # module.1, module.3).
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        where = _json_pointer.pointer(at, "$ref")
        _placed(lambda: resolve(schema["$ref"], root), where)
    if _json.type(schema.get("pattern")) == "string":
        where = _json_pointer.pointer(at, "pattern")
        _placed(lambda: _regular_expression.compile(schema["pattern"]), where)
    patterns_at = _json_pointer.pointer(at, "patternProperties")
    for pattern in schema.get("patternProperties", {}):
        where = _json_pointer.pointer(patterns_at, pattern)
        _placed(lambda: _regular_expression.compile(pattern), where)
    for where, child in _subschemas(schema, at):
        check(child, root, where)


def named(name: str, schema: dict[str, Any]) -> bool:
    # A member is named by properties or matched by a pattern of
    # patternProperties (JSON Schema Validation, 6.5.4. properties;
    # 6.5.5. patternProperties).
    return name in schema.get("properties", {}) or any(
        _regular_expression.search(pattern, name)
        for pattern in schema.get("patternProperties", {})
    )


def _applied(instance: Any, schema: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    # Each child location is paired with the subschema applied to it:
    # items and additionalItems apply to elements, and properties,
    # patternProperties and additionalProperties to member values (JSON
    # Schema Validation, 3.1. Applicability).
    found: list[tuple[Any, Any, Any]] = []
    if _json.type(instance) == "array":
        items = schema.get("items", True)
        rest = schema.get("additionalItems", True)
        for position, element in enumerate(instance):
            if not isinstance(items, list):
                found.append((position, element, items))
            elif position < len(items):
                found.append((position, element, items[position]))
            else:
                found.append((position, element, rest))
    if _json.type(instance) == "object":
        for name, member in instance.items():
            for child in _member_schemas(name, schema):
                found.append((name, member, child))
    return found


def _member_schemas(name: str, schema: dict[str, Any]) -> list[Any]:
    # A member's schemas are its property's and every matching
    # pattern's, else additionalProperties' (JSON Schema Validation,
    # 6.5.6. additionalProperties).
    found = [schema["properties"][name]] if name in schema.get("properties", {}) else []
    for pattern, child in schema.get("patternProperties", {}).items():
        if _regular_expression.search(pattern, name):
            found.append(child)
    if not named(name, schema):
        found.append(schema.get("additionalProperties", True))
    return found


def _subschemas(schema: dict[str, Any], at: str) -> list[tuple[str, Any]]:
    # A schema's subschemas, those it holds in definitions included,
    # each come with the pointer to it.
    found = [
        (_json_pointer.pointer(at, name), schema[name])
        for name in _SCHEMA
        if name in schema
    ]
    for name in _SCHEMAS:
        list_at = _json_pointer.pointer(at, name)
        for i, child in enumerate(schema.get(name, [])):
            found.append((_json_pointer.pointer(list_at, i), child))
    for name in (*_SCHEMA_OBJECTS, "dependencies"):
        object_at = _json_pointer.pointer(at, name)
        for key, child in schema.get(name, {}).items():
            if not isinstance(child, list):
                found.append((_json_pointer.pointer(object_at, key), child))
    items = schema.get("items", True)
    items_at = _json_pointer.pointer(at, "items")
    if isinstance(items, list):
        for i, child in enumerate(items):
            found.append((_json_pointer.pointer(items_at, i), child))
    else:
        found.append((items_at, items))
    return found


def _placed(checked: Callable[[], Any], where: str) -> None:
    # A check that fails is placed by the pointer to what it checked.
    try:
        checked()
    except ValueError as error:
        raise _json_pointer.PlacedError(str(error), where) from None


def _equal(one: Any, other: Any) -> bool:
    # Two instances are equal when of the same type and value, numbers
    # by their mathematical value (JSON Schema, 4.2.3. Instance
    # Equality).
    found = _json.type(one)
    if found != _json.type(other):
        return False
    if found == "number":
        return Decimal(one) == Decimal(other)
    if found == "array":
        return len(one) == len(other) and all(map(_equal, one, other))
    if found == "object":
        return one.keys() == other.keys() and all(
            _equal(one[name], other[name]) for name in one
        )
    return one == other


def _is(instance: Any, kind: str) -> bool:
    # An assertion on a type other than the instance's always succeeds
    # (JSON Schema Validation, 3.2.1. Assertions and Instance Primitive
    # Types).
    return _json.type(instance) == kind


def _type(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # The type integer matches any number with a zero fractional part
    # (JSON Schema Validation, 6.1.1. type).
    names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    found = _json.type(instance)
    if found == "number" and "integer" in names:
        return Decimal(instance) == Decimal(instance).to_integral_value()
    return found in names


def _enum(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return any(_equal(instance, element) for element in schema["enum"])


def _const(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return _equal(instance, schema["const"])


def _multiple_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not _is(instance, "number") or (
        Decimal(instance) % Decimal(schema["multipleOf"]) == 0
    )


def _bound(name: str, holds: Callable[[Decimal, Decimal], bool]) -> Callable:
    def keyword(instance: Any, schema: dict[str, Any], root: Any) -> bool:
        return not _is(instance, "number") or holds(
            Decimal(instance), Decimal(schema[name])
        )

    return keyword


def _length(name: str, holds: Callable[[int, Decimal], bool]) -> Callable:
    # A length counts a string's characters, and a count an array's
    # elements or an object's properties (JSON Schema Validation, 6.3.1.
    # maxLength; 6.4.3. maxItems; 6.5.1. maxProperties).
    def keyword(instance: Any, schema: dict[str, Any], root: Any) -> bool:
        kind = {"Length": "string", "Items": "array", "Properties": "object"}
        applies = next(kind[end] for end in kind if name.endswith(end))
        return not _is(instance, applies) or holds(len(instance), Decimal(schema[name]))

    return keyword


def _pattern(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # A pattern is not implicitly anchored (JSON Schema Validation,
    # 6.3.3. pattern).
    return not _is(instance, "string") or _regular_expression.search(
        schema["pattern"], instance
    )


def _items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # A schema applies to every element, and an array of schemas each to
    # the element at its position (JSON Schema Validation, 6.4.1.
    # items).
    if not _is(instance, "array"):
        return True
    items = schema["items"]
    if isinstance(items, list):
        return all(validates(e, s, root) for e, s in zip(instance, items))
    return all(validates(element, items, root) for element in instance)


def _additional_items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # additionalItems applies only where items is an array of schemas,
    # to the elements beyond it (JSON Schema Validation, 6.4.2.
    # additionalItems).
    items = schema.get("items")
    if not _is(instance, "array") or not isinstance(items, list):
        return True
    rest = instance[len(items) :]
    return all(validates(e, schema["additionalItems"], root) for e in rest)


def _unique_items(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    if schema["uniqueItems"] is not True or not _is(instance, "array"):
        return True
    return not any(
        _equal(instance[i], instance[j])
        for i in range(len(instance))
        for j in range(i + 1, len(instance))
    )


def _contains(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not _is(instance, "array") or any(
        validates(element, schema["contains"], root) for element in instance
    )


def _required(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not _is(instance, "object") or all(
        name in instance for name in schema["required"]
    )


def _properties(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not _is(instance, "object") or all(
        validates(instance[name], child, root)
        for name, child in schema["properties"].items()
        if name in instance
    )


def _pattern_properties(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # A member validates against every pattern its name matches (JSON
    # Schema Validation, 6.5.5. patternProperties).
    if not _is(instance, "object"):
        return True
    return all(
        validates(value, child, root)
        for pattern, child in schema["patternProperties"].items()
        for name, value in instance.items()
        if _regular_expression.search(pattern, name)
    )


def _additional_properties(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # additionalProperties applies to the members that properties and
    # patternProperties do not name (JSON Schema Validation, 6.5.6.
    # additionalProperties).
    if not _is(instance, "object"):
        return True
    return all(
        validates(value, schema["additionalProperties"], root)
        for name, value in instance.items()
        if not named(name, schema)
    )


def _dependencies(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # Where the instance holds the key, it holds the names the
    # dependency lists, or validates against its schema (JSON Schema
    # Validation, 6.5.7. dependencies).
    if not _is(instance, "object"):
        return True
    for key, dependency in schema["dependencies"].items():
        if key not in instance:
            continue
        if isinstance(dependency, list):
            if not all(name in instance for name in dependency):
                return False
        elif not validates(instance, dependency, root):
            return False
    return True


def _property_names(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not _is(instance, "object") or all(
        validates(name, schema["propertyNames"], root) for name in instance
    )


def _if(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    # then applies where the instance is valid against if, and else
    # where it is not; without if, they are ignored (JSON Schema
    # Validation, 6.6. Keywords for Applying Subschemas Conditionally).
    branch = "then" if validates(instance, schema["if"], root) else "else"
    return branch not in schema or validates(instance, schema[branch], root)


def _all_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return all(validates(instance, child, root) for child in schema["allOf"])


def _any_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return any(validates(instance, child, root) for child in schema["anyOf"])


def _one_of(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    found = [validates(instance, child, root) for child in schema["oneOf"]]
    return found.count(True) == 1


def _not(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return not validates(instance, schema["not"], root)


_KEYWORDS: dict[str, Callable[[Any, dict[str, Any], Any], bool]] = {
    "type": _type,
    "enum": _enum,
    "const": _const,
    "multipleOf": _multiple_of,
    "maximum": _bound("maximum", lambda value, limit: value <= limit),
    "exclusiveMaximum": _bound("exclusiveMaximum", lambda value, limit: value < limit),
    "minimum": _bound("minimum", lambda value, limit: value >= limit),
    "exclusiveMinimum": _bound("exclusiveMinimum", lambda value, limit: value > limit),
    "maxLength": _length("maxLength", lambda size, limit: size <= limit),
    "minLength": _length("minLength", lambda size, limit: size >= limit),
    "pattern": _pattern,
    "items": _items,
    "additionalItems": _additional_items,
    "maxItems": _length("maxItems", lambda size, limit: size <= limit),
    "minItems": _length("minItems", lambda size, limit: size >= limit),
    "uniqueItems": _unique_items,
    "contains": _contains,
    "maxProperties": _length("maxProperties", lambda size, limit: size <= limit),
    "minProperties": _length("minProperties", lambda size, limit: size >= limit),
    "required": _required,
    "properties": _properties,
    "patternProperties": _pattern_properties,
    "additionalProperties": _additional_properties,
    "dependencies": _dependencies,
    "propertyNames": _property_names,
    "if": _if,
    "allOf": _all_of,
    "anyOf": _any_of,
    "oneOf": _one_of,
    "not": _not,
}
