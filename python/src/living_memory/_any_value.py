"""How an AnyValue and the JSON value it maps to are converted."""

__all__ = [
    "convert",
    "is_double",
    "is_int64",
    "is_integer",
    "members",
    "represent",
    "values",
    "winner",
]

import base64
import math
from decimal import Decimal
from typing import Any

from living_memory import _fields, _protojson
from living_memory._json import Number, is_unicode

# An AnyValue's value is one of these members, each of its type
# (OTLP common.proto).
_MEMBERS = {
    "stringValue": "string",
    "boolValue": "bool",
    "intValue": "int64",
    "doubleValue": "double",
    "arrayValue": "ArrayValue",
    "kvlistValue": "KeyValueList",
    "bytesValue": "bytes",
    "stringValueStrindex": "int32",
}


def represent(value: Any) -> Any:
    # An AnyValue is read as the JSON value it maps to: a string as a
    # string, a boolean as a boolean, an int or a double as a number,
    # NaN and Infinity as those strings, bytes as their Base64 string,
    # an empty value as null, an array as an array and a key-value list
    # as an object (OTEL-COMMON, AnyValue representation for non-OTLP
    # protocols; spec › attributes.1). A string_value_strindex is read
    # as absent (spec › file.3).
    name, member = winner(value)
    if name in ("", "stringValueStrindex"):
        return None
    if name == "intValue":
        return Number(member)
    if name == "doubleValue":
        if not isinstance(member, Number) and member in _protojson.NON_FINITE:
            return str(member)
        return Number(member)
    if name == "arrayValue":
        return [represent(element) for element in values(member)]
    if name == "kvlistValue":
        return {
            _fields.text(pair.get("key")): represent(pair.get("value"))
            for pair in values(member)
            if isinstance(pair, dict)
        }
    return member


def convert(value: Any) -> dict[str, Any]:
    # A JSON value becomes an AnyValue: an object a kvlist_value and an
    # array an array_value, each converted in turn, a boolean a
    # bool_value, null an empty AnyValue, a number by its range, a valid
    # Unicode string a string_value, and any other string a bytes_value
    # holding its code units in order (OTEL-COMMON, Converting to
    # AnyValue; spec › event.3).
    if isinstance(value, dict):
        pairs = [{"key": name, "value": convert(v)} for name, v in value.items()]
        return {"kvlistValue": {"values": pairs}}
    if isinstance(value, list):
        return {"arrayValue": {"values": [convert(v) for v in value]}}
    if isinstance(value, bool):
        return {"boolValue": value}
    if value is None:
        return {}
    if isinstance(value, Number):
        return _number(value)
    if not is_unicode(value):
        data = value.encode("utf-8", "surrogatepass")
        return {"bytesValue": base64.b64encode(data).decode("ascii")}
    return {"stringValue": value}


def winner(value: Any) -> tuple[str, Any]:
    # Of the members of a oneof, the last is read, and a member that is
    # null is absent (PROTOJSON, Duplicate keys; Null values).
    found: tuple[str, Any] = ("", None)
    if isinstance(value, dict):
        for name, member in value.items():
            if name in _MEMBERS and member is not None:
                found = (name, member)
    return found


def members() -> list[tuple[str, str]]:
    # The members are given with their types, in the proto's order.
    return list(_MEMBERS.items())


def values(value: Any) -> list[Any]:
    # An ArrayValue and a KeyValueList hold their values in a list
    # (OTLP common.proto).
    if not isinstance(value, dict):
        return []
    found = value.get("values")
    return found if isinstance(found, list) else []


def is_int64(value: Any) -> bool:
    # A number with a zero fractional part within the 64-bit signed
    # range is an int_value (OTEL-COMMON, Integer Values;
    # JSON-SCHEMA-07, validation 6.1.1. type).
    return isinstance(value, Number) and _protojson.fits(value, "int64")


def is_double(value: Any) -> bool:
    # A number within the range of an IEEE 754 64-bit double
    # (OTEL-COMMON, Floating Point Values).
    return isinstance(value, Number) and not math.isinf(float(value))


def is_integer(value: Number) -> bool:
    number = Decimal(value)
    return number == number.to_integral()


def _number(value: Number) -> dict[str, Any]:
    # An int_value is written as the decimal string of its integer
    # value, so 40.0 and 4e1 are both 40; beyond the range of its type,
    # a number is a string_value (OTLP, JSON Protobuf Encoding;
    # OTEL-COMMON, Integer Values and Floating Point Values).
    if is_int64(value):
        return {"intValue": str(int(Decimal(value)))}
    if is_integer(value) or not is_double(value):
        return {"stringValue": str(value)}
    return {"doubleValue": value}
