"""Whether a JSON value validates against a JSON Schema of draft-07."""

from __future__ import annotations

__all__ = [
    "compile_pattern",
    "covers",
    "equal",
    "locate",
    "resolve",
    "search",
    "subschemas",
    "validates",
]

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from living_memory import _json, _json_pointer

# These keywords hold a schema, a list of schemas, or an object of
# schemas (JSON Schema Validation, 3.1. Applicability; 9. Schema Re-Use
# With "definitions").
_SCHEMA = ("additionalItems", "contains", "propertyNames", "not")
_SCHEMA += ("if", "then", "else", "additionalProperties")
_SCHEMAS = ("allOf", "anyOf", "oneOf")
_SCHEMA_OBJECTS = ("properties", "patternProperties", "definitions")

# Each simple quantifier schema authors should limit themselves to is
# given by the least and most occurrences it allows; it, and each range
# quantifier, may be lazy (JSON Schema Validation, 4.3. Regular
# Expressions).
_QUANTIFIERS = {"*": (0, math.inf), "+": (1, math.inf), "?": (0, 1)}
# These tokens are outside the subset.
_OUTSIDE = ".\\]}"
_DIGITS = "0123456789"


def validates(instance: Any, schema: Any, root: Any) -> bool:
    # A schema is true, false or an object; a $ref is used in its place,
    # its other members ignored; every assertion applies, and format and
    # the content keywords are not asserted (JSON Schema, 4.3.1. JSON
    # Schema Values and Keywords; 8.3. Schema References With "$ref";
    # spec › module.2, value.3).
    if schema is True or schema is False:
        return schema
    if "$ref" in schema:
        return validates(instance, resolve(schema["$ref"], root)[0], root)
    return all(
        keyword(instance, schema, root)
        for name, keyword in _KEYWORDS.items()
        if name in schema
    )


def locate(instance: Any, schema: Any, root: Any) -> str:
    # For an instance that does not validate, the place named is the
    # deepest instance location where an assertion fails: a child
    # location whose subschema fails, else a failing subschema allOf
    # applies here, else this location (JSON Schema Validation, 3.1.
    # Applicability; spec › value.3).
    def deepest(instance: Any, schema: Any, at: str) -> str:
        if schema is False:
            return at
        if "$ref" in schema:
            return deepest(instance, resolve(schema["$ref"], root)[0], at)
        for token, child, child_schema in _applied(instance, schema):
            if not validates(child, child_schema, root):
                child_at = _json_pointer.pointer(at, token)
                return deepest(child, child_schema, child_at)
        for child_schema in schema.get("allOf", []):
            if not validates(instance, child_schema, root):
                return deepest(instance, child_schema, at)
        return at

    return deepest(instance, schema, "")


def resolve(reference: str, root: Any) -> tuple[Any, str]:
    # A reference resolves within the one schema given, to the schema
    # and its pointer (JSON Schema, 8.3.1. Loading a referenced schema;
    # 8.3.2. Dereferencing; spec › module.2).
    base, _, fragment = reference.partition("#")
    if base:
        raise ValueError(f"$ref {reference!r} is outside the supplied schema")
    try:
        return _json_pointer.evaluate(root, fragment), fragment
    except (KeyError, IndexError, ValueError, TypeError):
        raise ValueError(f"$ref {reference!r} resolves to no schema") from None


def covers(name: str, schema: dict[str, Any]) -> bool:
    # A schema covers a member that its properties names or a pattern
    # of its patternProperties matches (JSON Schema Validation, 6.5.4.
    # properties; 6.5.5. patternProperties; spec › relation.5).
    return name in schema.get("properties", {}) or any(
        search(pattern, name) for pattern in schema.get("patternProperties", {})
    )


def subschemas(schema: Any, at: str) -> list[tuple[str, Any]]:
    # A schema's subschemas, those it holds in definitions included,
    # each come with the pointer to it.
    if not isinstance(schema, dict):
        return []
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
    if "items" in schema:
        items = schema["items"]
        items_at = _json_pointer.pointer(at, "items")
        if isinstance(items, list):
            for i, child in enumerate(items):
                found.append((_json_pointer.pointer(items_at, i), child))
        else:
            found.append((items_at, items))
    return found


def compile_pattern(pattern: str) -> _Group:
    # A pattern holds only individual characters, simple and
    # complemented character classes and ranges, the quantifiers, the
    # anchors ^ and $, and simple grouping and alternation; any other
    # token is refused (JSON Schema Validation, 4.3. Regular
    # Expressions; spec › module.1).
    group, at = _alternation(pattern, 0)
    if at < len(pattern):
        raise ValueError(f"an unmatched ) in {pattern!r}")
    return group


def search(pattern: str, text: str) -> bool:
    # A pattern matches a string where it matches from any position in
    # it, not implicitly anchored at either end (JSON Schema Validation,
    # 4.3. Regular Expressions; 6.3.3. pattern).
    group = compile_pattern(pattern)
    return bool(group.step(text, frozenset(range(len(text) + 1))))


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
        if search(pattern, name):
            found.append(child)
    if not covers(name, schema):
        found.append(schema.get("additionalProperties", True))
    return found


def equal(one: Any, other: Any) -> bool:
    # Two instances are equal when of the same type and value, numbers
    # by their mathematical value (JSON Schema, 4.2.3. Instance
    # Equality).
    found = _json.type(one)
    if found != _json.type(other):
        return False
    if found == "number":
        return Decimal(one) == Decimal(other)
    if found == "array":
        return len(one) == len(other) and all(map(equal, one, other))
    if found == "object":
        return one.keys() == other.keys() and all(
            equal(one[name], other[name]) for name in one
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
    return any(equal(instance, element) for element in schema["enum"])


def _const(instance: Any, schema: dict[str, Any], root: Any) -> bool:
    return equal(instance, schema["const"])


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
    return not _is(instance, "string") or search(schema["pattern"], instance)


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
        equal(instance[i], instance[j])
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
        if search(pattern, name)
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
        if not covers(name, schema)
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


def _alternation(pattern: str, at: int) -> tuple[_Group, int]:
    sequences = []
    sequence, at = _sequence(pattern, at)
    sequences.append(sequence)
    while pattern[at : at + 1] == "|":
        sequence, at = _sequence(pattern, at + 1)
        sequences.append(sequence)
    return _Group(tuple(sequences)), at


def _sequence(pattern: str, at: int) -> tuple[tuple[object, ...], int]:
    # A sequence runs to an alternation or the end of its group; a
    # quantifier follows a character, a class or a group.
    nodes: list[object] = []
    while at < len(pattern) and pattern[at] not in "|)":
        node, at = _atom(pattern, at)
        if pattern[at : at + 1] in ("*", "+", "?", "{"):
            if isinstance(node, (_Start, _End)):
                raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
            least, most, at = _bounds(pattern, at)
            if pattern[at : at + 1] == "?":
                at += 1
            node = _Repeat(node, least, most)
        nodes.append(node)
    return tuple(nodes), at


def _atom(pattern: str, at: int) -> tuple[object, int]:
    char = pattern[at]
    if char == "(":
        if pattern.startswith("(?", at):
            raise ValueError(f"not a simple group in {pattern!r}")
        group, end = _alternation(pattern, at + 1)
        if pattern[end : end + 1] != ")":
            raise ValueError(f"an unmatched ( in {pattern!r}")
        return group, end + 1
    if char == "[":
        return _class(pattern, at)
    if char == "^":
        return _Start(), at + 1
    if char == "$":
        return _End(), at + 1
    if char in _OUTSIDE:
        raise ValueError(f"{char!r} is not among the tokens of {pattern!r}")
    if char in _QUANTIFIERS or char == "{":
        raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
    return _Char(char), at + 1


def _bounds(pattern: str, at: int) -> tuple[int, float, int]:
    # A quantifier is a simple one, or {x}, {x,y} or {x,}, where x and
    # y are decimal digits.
    char = pattern[at]
    if char in _QUANTIFIERS:
        least, most = _QUANTIFIERS[char]
        return least, most, at + 1
    end = pattern.find("}", at)
    low, comma, high = pattern[at + 1 : end].partition(",")
    if end == -1 or not _decimal(low) or (high and not _decimal(high)):
        raise ValueError(f"a quantifier with nothing to repeat in {pattern!r}")
    least = int(low)
    most = int(high) if high else math.inf if comma else least
    if most < least:
        raise ValueError(f"a range quantifier out of order in {pattern!r}")
    return least, most, end + 1


def _class(pattern: str, at: int) -> tuple[_Class, int]:
    # A class is [abc], [a-z], [^abc] or [^a-z], holding no escape and
    # no class; a '-' first or last in the class is itself.
    end = pattern.find("]", at + 1)
    inside = pattern[at + 1 : end] if end != -1 else ""
    complemented = inside.startswith("^")
    members = inside[1:] if complemented else inside
    if end == -1 or not members or "\\" in members or "[" in members:
        raise ValueError(f"not a simple character class in {pattern!r}")
    ranges = []
    i = 0
    while i < len(members):
        if i + 2 < len(members) and members[i + 1] == "-":
            if members[i] > members[i + 2]:
                raise ValueError(f"a range out of order in {pattern!r}")
            ranges.append((members[i], members[i + 2]))
            i += 3
        else:
            ranges.append((members[i], members[i]))
            i += 1
    return _Class(complemented, tuple(ranges)), end + 1


def _decimal(digits: str) -> bool:
    return bool(digits) and all(digit in _DIGITS for digit in digits)


# Each node of a pattern takes the positions a match may have reached in
# the text and returns the positions it may reach after the node.


@dataclass(frozen=True)
class _Char:
    char: str

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return frozenset(p + 1 for p in positions if text[p : p + 1] == self.char)


@dataclass(frozen=True)
class _Class:
    complemented: bool
    ranges: tuple[tuple[str, str], ...]

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return frozenset(
            p + 1 for p in positions if p < len(text) and self._holds(text[p])
        )

    def _holds(self, char: str) -> bool:
        inside = any(low <= char <= high for low, high in self.ranges)
        return inside != self.complemented


@dataclass(frozen=True)
class _Start:
    # ^ matches at the beginning of input.
    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return positions & {0}


@dataclass(frozen=True)
class _End:
    # $ matches at the end of input, and not before a final line break.
    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        return positions & {len(text)}


@dataclass(frozen=True)
class _Group:
    # A group matches where any of its alternatives does, each a
    # sequence of nodes matched one after another.
    sequences: tuple[tuple[object, ...], ...]

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        reached: frozenset[int] = frozenset()
        for sequence in self.sequences:
            current = positions
            for node in sequence:
                current = node.step(text, current)
            reached |= current
        return reached


@dataclass(frozen=True)
class _Repeat:
    # A quantified node matches from least to most times; a lazy form
    # reaches the same positions, and so matches where its greedy form
    # does.
    node: object
    least: int
    most: float

    def step(self, text: str, positions: frozenset[int]) -> frozenset[int]:
        current = positions
        reached = positions if self.least == 0 else frozenset()
        count = 0
        while current and count < self.most:
            current = self.node.step(text, current)
            count += 1
            if count >= self.least:
                # Once past least, positions already reached lead only
                # to positions already reached.
                if current <= reached:
                    break
                reached |= current
        return reached
