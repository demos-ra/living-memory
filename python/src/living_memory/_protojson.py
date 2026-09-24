"""How a simple value of an OTLP message is written in JSON."""

__all__ = ["valid", "NON_FINITE"]

import re
import string
from decimal import Decimal
from typing import Any

from living_memory._json import Number

# OTLP, JSON Protobuf Encoding: hex in all OTLP messages.
_HEX = frozenset({"traceId", "spanId"})
# PROTOJSON, Representation of each type.
NON_FINITE = frozenset({"NaN", "Infinity", "-Infinity"})
# RFC 8259, Section 6.
_NUMERAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
# PROTOJSON, Out of range numeric values.
_RANGES = {
    "int32": (-(2**31), 2**31 - 1),
    "enum": (-(2**31), 2**31 - 1),
    "uint32": (0, 2**32 - 1),
    "fixed32": (0, 2**32 - 1),
    "sint32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "sfixed64": (-(2**63), 2**63 - 1),
    "fixed64": (0, 2**64 - 1),
    "uint64": (0, 2**64 - 1),
}
# RFC 4648, Sections 4 and 5.
_BASE64 = frozenset(string.ascii_letters + string.digits + "+/-_")


def valid(value: Any, proto: str, name: str) -> bool:
    # PROTOJSON, Representation of each type; OTLP, JSON Protobuf
    # Encoding: enums are integers.
    if proto == "string":
        return isinstance(value, str) and not isinstance(value, Number)
    if proto == "bool":
        return isinstance(value, bool)
    if proto == "bytes":
        return _is_hex(value) if name in _HEX else _is_base64(value)
    if proto == "double":
        return isinstance(value, Number) or (
            isinstance(value, str) and (value in NON_FINITE or _is_numeral(value))
        )
    if proto == "enum":
        return isinstance(value, Number) and _fits(value, proto)
    return isinstance(value, str) and _is_numeral(value) and _fits(value, proto)


def _is_hex(value: Any) -> bool:
    # RFC 4648, Section 8.
    return (
        isinstance(value, str)
        and not isinstance(value, Number)
        and len(value) % 2 == 0
        and all(char in string.hexdigits for char in value)
    )


def _is_base64(value: Any) -> bool:
    # PROTOJSON: standard or URL-safe, with or without padding.
    if not isinstance(value, str) or isinstance(value, Number):
        return False
    data = value.rstrip("=")
    return (
        len(value) - len(data) <= 2
        and len(data) % 4 != 1
        and all(char in _BASE64 for char in data)
    )


def _is_numeral(value: str) -> bool:
    return _NUMERAL.fullmatch(value) is not None


def _fits(value: str, proto: str) -> bool:
    # PROTOJSON, Out of range numeric values.
    number = Decimal(value)
    low, high = _RANGES[proto]
    return number == number.to_integral() and low <= number <= high
