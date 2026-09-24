"""How a simple value of an OTLP message is written in JSON."""

__all__ = ["fits", "valid", "NON_FINITE"]

import re
import string
from decimal import Decimal
from typing import Any

from living_memory._json import Number

# traceId and spanId are hex strings in all OTLP messages, where
# ProtoJSON writes bytes as base64 (OTLP, JSON Protobuf Encoding).
_HEX = frozenset({"traceId", "spanId"})
# A double may be one of these strings (PROTOJSON, Representation of
# each type).
NON_FINITE = frozenset({"NaN", "Infinity", "-Infinity"})
# A number: an optional minus sign, an integer part without leading
# zeros, and an optional fraction and exponent (RFC8259, 6. Numbers).
_NUMERAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
# Each integer type has this range (PROTOJSON, Out of range numeric
# values).
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
# Base64 is read in its alphabet and in the URL and filename safe one
# (RFC4648, 4. Base 64 Encoding; 5. Base 64 Encoding with URL and
# Filename Safe Alphabet).
_BASE64 = frozenset(string.ascii_letters + string.digits + "+/-_")


def valid(value: Any, proto: str, name: str) -> bool:
    # Each type has its JSON form, and an enum is an integer, never its
    # name (PROTOJSON, Representation of each type; OTLP, JSON Protobuf
    # Encoding).
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
        return isinstance(value, Number) and fits(value, proto)
    return isinstance(value, str) and _is_numeral(value) and fits(value, proto)


def fits(value: str, proto: str) -> bool:
    # A number fits an integer type when it has no fractional part and
    # is within the type's range (PROTOJSON, Out of range numeric
    # values).
    number = Decimal(value)
    low, high = _RANGES[proto]
    return number == number.to_integral() and low <= number <= high


def _is_hex(value: Any) -> bool:
    # Hex encodes each octet as two characters, in either case (RFC4648,
    # 8. Base 16 Encoding).
    return (
        isinstance(value, str)
        and not isinstance(value, Number)
        and len(value) % 2 == 0
        and all(char in string.hexdigits for char in value)
    )


def _is_base64(value: Any) -> bool:
    # Bytes are standard or URL-safe base64, with or without padding
    # (PROTOJSON, Representation of each type); characters outside the
    # alphabet are rejected, and a padded final unit is two characters
    # and "==" or three and "=" (RFC4648, 3.3. Interpretation of
    # Non-Alphabet Characters in Encoded Data; 4. Base 64 Encoding).
    if not isinstance(value, str) or isinstance(value, Number):
        return False
    data = value.rstrip("=")
    padding = len(value) - len(data)
    if not all(char in _BASE64 for char in data):
        return False
    if padding == 0:
        return len(data) % 4 != 1
    return padding <= 2 and len(data) % 4 + padding == 4


def _is_numeral(value: str) -> bool:
    return _NUMERAL.fullmatch(value) is not None
