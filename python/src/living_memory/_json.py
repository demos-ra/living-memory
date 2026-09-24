"""How a JSON text is read and written, and the type of a JSON value."""

__all__ = [
    "Number",
    "array",
    "decode",
    "decode_all_pairs",
    "decode_strict",
    "encode",
    "is_unicode",
    "replace_unpaired",
    "type",
]

import json
import re
from typing import Any


class Number(str):
    # A number is kept as the text the input wrote, so that no digit is
    # lost to a machine number (RFC8259, 6. Numbers; spec › node.3).
    pass


# A UTF-16 surrogate encodes no Unicode character on its own; a pair
# written as escapes is read as the one character it encodes
# (RFC8259, 7. Strings; 8.2. Unicode Characters).
_SURROGATE = re.compile("[\ud800-\udfff]")


def decode(document: str) -> Any:
    # Of members sharing a name the last is read, as ProtoJSON's parsers
    # do (PROTOJSON, Duplicate keys; spec › file.3).
    return json.loads(
        document,
        parse_int=Number,
        parse_float=Number,
        parse_constant=_refuse_constant,
        object_pairs_hook=_keep_last_member,
    )


def decode_all_pairs(document: str) -> Any:
    # Members sharing a name are one member holding every value, in the
    # order written, as OpenTelemetry converts non-unique keys
    # (OTEL-COMMON, Associative Arrays With Non-Unique Keys; spec ›
    # event.4).
    return json.loads(
        document,
        parse_int=Number,
        parse_float=Number,
        parse_constant=_refuse_constant,
        object_pairs_hook=_keep_every_member,
    )


def decode_strict(document: str) -> Any:
    # Strict JSON, read to be written back, keeps its numbers as
    # numbers; NaN and Infinity are refused (RFC8259, 6. Numbers).
    return json.loads(document, parse_constant=_refuse_constant)


def encode(value: Any) -> str:
    # A Number is written as its own text, and every other value as
    # RFC8259 writes it, so that what was read is written unchanged.
    if isinstance(value, Number):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(encode(element) for element in value) + "]"
    members = (f"{encode(name)}:{encode(member)}" for name, member in value.items())
    return "{" + ",".join(members) + "}"


def is_unicode(text: str) -> bool:
    # A string is a valid Unicode sequence when it holds no unpaired
    # surrogate (RFC8259, 8.2. Unicode Characters).
    return _SURROGATE.search(text) is None


def replace_unpaired(value: Any) -> Any:
    # Every unpaired surrogate in a value's strings, names included, is
    # replaced with U+FFFD, as an OTLP decoder replaces what is not
    # valid UTF-8 (OTLP, UTF-8 String Handling; spec › file.1).
    if isinstance(value, dict):
        return {replace_unpaired(k): replace_unpaired(v) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_unpaired(element) for element in value]
    if isinstance(value, str) and not isinstance(value, Number):
        return _SURROGATE.sub("\ufffd", value)
    return value


def array(value: Any) -> list[Any]:
    # An array's elements are read, and any other value has none.
    return value if isinstance(value, list) else []


def type(value: Any) -> str:
    # The types are those of JSON Schema draft-07 (JSON-SCHEMA-07,
    # validation 6.1.1. type; spec › node.1).
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (Number, int, float)):
        return "number"
    return "string"


def _refuse_constant(name: str) -> Any:
    # NaN and Infinity are not JSON values (RFC8259, 6. Numbers).
    raise ValueError(f"{name} is not a JSON value")


def _keep_last_member(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for name, value in pairs:
        members.pop(name, None)
        members[name] = value
    return members


def _keep_every_member(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for name, value in pairs:
        grouped.setdefault(name, []).append(value)
    return {
        name: values[0] if len(values) == 1 else values
        for name, values in grouped.items()
    }
