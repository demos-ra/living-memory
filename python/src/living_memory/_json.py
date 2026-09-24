"""How a JSON text is read, and the type of a JSON value."""

__all__ = ["Number", "decode", "type"]

import json
from typing import Any


class Number(str):
    # spec › node.3; RFC 8259, Section 6.
    pass


def decode(document: str) -> Any:
    return json.loads(
        document,
        parse_int=Number,
        parse_float=Number,
        parse_constant=_refuse_constant,
        object_pairs_hook=_keep_last_member,
    )


def type(value: Any) -> str:
    # JSON-SCHEMA-07, validation 6.1.1; spec › node.1.
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
    # RFC 8259, Section 6.
    raise ValueError(f"{name} is not a JSON value")


def _keep_last_member(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # PROTOJSON, Duplicate keys; spec › file.3.
    members: dict[str, Any] = {}
    for name, value in pairs:
        members.pop(name, None)
        members[name] = value
    return members
