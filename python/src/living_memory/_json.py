"""How JSON texts are read, JSON strings written, and values typed."""

__all__ = [
    "HIGH_SURROGATES",
    "LOW_SURROGATES",
    "Number",
    "TYPES",
    "decode",
    "encode",
    "encode_string",
    "type",
]

from typing import Any

from living_memory import _utf_8
from living_memory._json_pointer import PlacedError, pointer

# The data model has six primitive types (JSON Schema, 4.2.1. Instance
# Data Model).
TYPES = ("null", "boolean", "object", "array", "number", "string")

# The UTF-16 surrogates are high, then low; a pair escapes a character
# beyond the Basic Multilingual Plane, and an unpaired one encodes no
# Unicode character (RFC 8259, 7. Strings; 8.2. Unicode Characters).
HIGH_SURROGATES = range(0xD800, 0xDC00)
LOW_SURROGATES = range(0xDC00, 0xE000)

# These are insignificant whitespace, the literal names, and the
# two-character escapes of a string (RFC 8259, 2. JSON Grammar; 3.
# Values; 7. Strings).
_WHITESPACE = " \t\n\r"
_LITERALS = {"false": False, "null": None, "true": True}
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS = "0123456789abcdefABCDEF"
_DIGITS = "0123456789"
# U+10000 is the first code point beyond the Basic Multilingual Plane.
_BEYOND_BMP = 0x10000

# A reader returns the value, the index after it, and the start and
# pointer of every object within it whose names repeat.
_Read = tuple[Any, int, list[tuple[int, str]]]


class Number(str):
    # A number is kept as the text written, since an implementation may
    # limit the range and precision of numbers (RFC 8259, 6. Numbers;
    # spec › value.2).
    pass


def decode(data: bytes) -> Any:
    # A JSON text is a value between insignificant whitespace, in UTF-8;
    # members are read in the order written. A text that is not one
    # fails whole, and so is placed at ""; one whose names repeat is
    # placed at the first such object in the order written (RFC 8259,
    # 2. JSON Grammar; 4. Objects; 8.1. Character Encoding; spec ›
    # value.1-3).
    try:
        text = _utf_8.decode(data)
        value, end, repeated = _value(text, _whitespace(text, 0), "")
        end = _whitespace(text, end)
        if end != len(text):
            raise ValueError(f"text after the value at character {end}")
    except ValueError as error:
        raise PlacedError(f"not a JSON text: {error}", "") from None
    if repeated:
        raise PlacedError("a name is repeated", min(repeated)[1])
    return value


def encode(value: Any, indent: int = 0) -> bytes:
    # A value is written as a JSON text in UTF-8: an object's members in
    # their order, an array's elements, a number as its text as written,
    # a string escaped, and the literal names; with an indent, each
    # member and element on a line of its own, indented by its depth,
    # as insignificant whitespace (RFC 8259, 2. JSON Grammar; 3. Values;
    # 4. Objects; 5. Arrays; 6. Numbers; 7. Strings; 8.1. Character
    # Encoding).
    return _text(value, indent, 1).encode("utf-8")


def encode_string(value: str) -> str:
    # A string is written between quotation marks, the quotation mark,
    # the reverse solidus and the control characters escaped (RFC 8259,
    # 7. Strings); an unpaired surrogate, which is not text, is escaped
    # too (8.2. Unicode Characters).
    written = {code: f"\\{name}" for name, code in _ESCAPES.items() if name != "/"}
    return '"' + "".join(_escaped(char, written) for char in value) + '"'


def type(value: Any) -> str:
    # A value has one of the six primitive types; an integer is a
    # number (JSON Schema, 4.2.1. Instance Data Model).
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, Number):
        return "number"
    return "string"


def _text(value: Any, indent: int, depth: int) -> str:
    found = type(value)
    if found in ("object", "array") and value:
        inner = "\n" + " " * (indent * depth) if indent else ""
        outer = "\n" + " " * (indent * (depth - 1)) if indent else ""
        colon = ": " if indent else ":"
        if found == "object":
            parts = [
                f"{encode_string(k)}{colon}{_text(v, indent, depth + 1)}"
                for k, v in value.items()
            ]
            ends = "{}"
        else:
            parts = [_text(element, indent, depth + 1) for element in value]
            ends = "[]"
        return ends[0] + inner + ("," + inner).join(parts) + outer + ends[1]
    if found == "object":
        return "{}"
    if found == "array":
        return "[]"
    if found == "string":
        return encode_string(value)
    if found == "number":
        return str(value)
    return {True: "true", False: "false", None: "null"}[value]


def _whitespace(text: str, at: int) -> int:
    while at < len(text) and text[at] in _WHITESPACE:
        at += 1
    return at


def _value(text: str, at: int, where: str) -> _Read:
    # A value is an object, an array, a number, a string, or one of the
    # three literal names (RFC 8259, 3. Values).
    char = text[at : at + 1]
    if char == "{":
        return _object(text, at, where)
    if char == "[":
        return _array(text, at, where)
    if char == '"':
        string, end = _string(text, at)
        return string, end, []
    if char and char in "-" + _DIGITS:
        number, end = _number(text, at)
        return number, end, []
    for name, literal in _LITERALS.items():
        if text.startswith(name, at):
            return literal, at + len(name), []
    raise ValueError(f"no JSON value at character {at}")


def _object(text: str, start: int, where: str) -> _Read:
    # An object is zero or more members between curly brackets, a name
    # and a value each, separated by commas; its names should be unique
    # (RFC 8259, 4. Objects).
    members: dict[str, Any] = {}
    repeated: list[tuple[int, str]] = []
    repeats = False
    at = _whitespace(text, start + 1)
    if text[at : at + 1] == "}":
        return members, at + 1, repeated
    while True:
        if text[at : at + 1] != '"':
            raise ValueError(f"no name at character {at}")
        name, at = _string(text, at)
        at = _whitespace(text, at)
        if text[at : at + 1] != ":":
            raise ValueError(f"no colon at character {at}")
        at = _whitespace(text, at + 1)
        value, at, inner = _value(text, at, pointer(where, name))
        repeated += inner
        repeats = repeats or name in members
        members[name] = value
        at = _whitespace(text, at)
        if text[at : at + 1] == "}":
            if repeats:
                repeated.append((start, where))
            return members, at + 1, repeated
        if text[at : at + 1] != ",":
            raise ValueError(f"no comma or end of object at character {at}")
        at = _whitespace(text, at + 1)


def _array(text: str, start: int, where: str) -> _Read:
    # An array is zero or more values between square brackets,
    # separated by commas (RFC 8259, 5. Arrays).
    elements: list[Any] = []
    repeated: list[tuple[int, str]] = []
    at = _whitespace(text, start + 1)
    if text[at : at + 1] == "]":
        return elements, at + 1, repeated
    while True:
        value, at, inner = _value(text, at, pointer(where, len(elements)))
        repeated += inner
        elements.append(value)
        at = _whitespace(text, at)
        if text[at : at + 1] == "]":
            return elements, at + 1, repeated
        if text[at : at + 1] != ",":
            raise ValueError(f"no comma or end of array at character {at}")
        at = _whitespace(text, at + 1)


def _string(text: str, start: int) -> tuple[str, int]:
    # A string is characters between quotation marks; the quotation
    # mark, the reverse solidus and the control characters are escaped,
    # and a character outside the Basic Multilingual Plane may be
    # escaped as its UTF-16 surrogate pair (RFC 8259, 7. Strings).
    chars = []
    at = start + 1
    while True:
        char = text[at : at + 1]
        if not char:
            raise ValueError(f"the string at character {start} does not end")
        if char == '"':
            return "".join(chars), at + 1
        if char < " ":
            raise ValueError(f"a control character unescaped at character {at}")
        if char != "\\":
            chars.append(char)
            at += 1
            continue
        escape = text[at + 1 : at + 2]
        if escape and escape in _ESCAPES:
            chars.append(_ESCAPES[escape])
            at += 2
            continue
        code = _hex(text, at)
        at += 6
        if code in HIGH_SURROGATES and text.startswith("\\u", at):
            low = _hex(text, at)
            if low in LOW_SURROGATES:
                code = _pair(code, low)
                at += 6
        chars.append(chr(code))


def _pair(high: int, low: int) -> int:
    # A surrogate pair encodes a code point above the Basic Multilingual
    # Plane: ten bits from each surrogate, added to U+10000.
    high_bits = high - HIGH_SURROGATES.start
    low_bits = low - LOW_SURROGATES.start
    return _BEYOND_BMP + (high_bits << 10) + low_bits


def _hex(text: str, at: int) -> int:
    # The escape \uXXXX is four hexadecimal digits that encode a code
    # point (RFC 8259, 7. Strings).
    digits = text[at + 2 : at + 6]
    if text[at + 1 : at + 2] != "u" or len(digits) < 4:
        raise ValueError(f"no escape at character {at}")
    if not all(digit in _HEX_DIGITS for digit in digits):
        raise ValueError(f"no escape at character {at}")
    return int(digits, 16)


def _number(text: str, start: int) -> tuple[Number, int]:
    # A number is an optional minus sign, an integer part without
    # leading zeros, then an optional fraction and exponent (RFC 8259,
    # 6. Numbers).
    at = start + 1 if text[start] == "-" else start
    if text[at : at + 1] == "0":
        at += 1
    else:
        at = _digits(text, at)
    if text[at : at + 1] == ".":
        at = _digits(text, at + 1)
    if text[at : at + 1] in ("e", "E"):
        at += 1
        if text[at : at + 1] in ("+", "-"):
            at += 1
        at = _digits(text, at)
    return Number(text[start:at]), at


def _digits(text: str, at: int) -> int:
    # A run of digits holds at least one digit.
    end = at
    while end < len(text) and text[end] in _DIGITS:
        end += 1
    if end == at:
        raise ValueError(f"no digit at character {at}")
    return end


def _escaped(char: str, written: dict[str, str]) -> str:
    if char in written:
        return written[char]
    code = ord(char)
    if char < " " or code in HIGH_SURROGATES or code in LOW_SURROGATES:
        return f"\\u{code:04x}"
    return char
