"""How JSON texts are read, JSON strings written, and values typed."""

__all__ = [
    "HIGH_SURROGATES",
    "LOW_SURROGATES",
    "Number",
    "TYPES",
    "decode",
    "encode",
    "encode_string",
    "primitive_type",
]

import re
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
# The characters a string holds unescaped: every one but the quotation
# mark, the reverse solidus and the control characters, U+0000 to U+001F
# (RFC 8259, 7. Strings: unescaped = %x20-21 / %x23-5B / %x5D-10FFFF).
_UNESCAPED = re.compile(r'[^"\\\x00-\x1f]+')
_HEX_DIGITS = "0123456789abcdefABCDEF"
_DIGITS = "0123456789"
# U+10000 is the first code point beyond the Basic Multilingual Plane.
_BEYOND_BMP = 0x10000
# An object is written between curly brackets, an array between square
# ones (RFC 8259, 4. Objects; 5. Arrays).
_BRACKETS = {"{": "}", "[": "]"}


class Number(str):
    # A number is kept as the text written, since an implementation may
    # limit the range and precision of numbers (RFC 8259, 6. Numbers;
    # spec › value.6).
    pass


class _Open:
    # An object or an array still being read: its container, its place,
    # where it starts, the name of the member being read, and whether a
    # name has come twice.
    def __init__(self, bracket: str, where: str, start: int) -> None:
        self.close = _BRACKETS[bracket]
        self.container: Any = {} if bracket == "{" else []
        self.where = where
        self.start = start
        self.name = ""
        self.repeats = False

    def place(self) -> str:
        # The pointer of the value being read within this container.
        token = self.name if self.close == "}" else len(self.container)
        return pointer(self.where, token)

    def add(self, value: Any) -> None:
        if self.close == "]":
            self.container.append(value)
            return
        self.repeats = self.repeats or self.name in self.container
        self.container[self.name] = value


def decode(data: bytes) -> Any:
    # A JSON text is a value between insignificant whitespace, in UTF-8;
    # members are read in the order written. A text that is not one
    # fails whole, and so is placed at ""; one whose names repeat is
    # placed at the first such object in the order written (RFC 8259,
    # 2. JSON Grammar; 4. Objects; 8.1. Character Encoding; spec ›
    # value.3, value.4, value.7, value.8, value.10).
    try:
        text = _utf_8.decode(data)
        value, end, repeated = _read(text)
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
    return _text(value, indent).encode("utf-8")


def encode_string(value: str) -> str:
    # A string is written between quotation marks, the quotation mark,
    # the reverse solidus and the control characters escaped (RFC 8259,
    # 7. Strings); an unpaired surrogate, which is not text, is escaped
    # too (8.2. Unicode Characters).
    written = {code: f"\\{name}" for name, code in _ESCAPES.items() if name != "/"}
    return '"' + "".join(_escaped(char, written) for char in value) + '"'


def primitive_type(value: Any) -> str:
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


def _read(text: str) -> tuple[Any, int, list[tuple[int, str]]]:
    # A value is read left to right, each object and array still open
    # kept in a list, so any depth of nesting is read (RFC 8259, 3.
    # Values; spec › value.3). It returns the value, the index after it,
    # and the start and pointer of every object whose names repeat.
    repeated: list[tuple[int, str]] = []
    opened: list[_Open] = []
    at = _whitespace(text, 0)
    while True:
        where = opened[-1].place() if opened else ""
        char = text[at : at + 1]
        if char in _BRACKETS:
            container = _Open(char, where, at)
            at = _whitespace(text, at + 1)
            if text[at : at + 1] != container.close:
                opened.append(container)
                at = _first(text, at, container)
                continue
            value, at = container.container, at + 1
        else:
            value, at = _scalar(text, at)
        # A value read is added to the container that holds it, and each
        # container it completes to the one that holds that.
        while opened:
            holder = opened[-1]
            holder.add(value)
            at = _whitespace(text, at)
            if text[at : at + 1] == ",":
                at = _first(text, _whitespace(text, at + 1), holder)
                break
            if text[at : at + 1] != holder.close:
                raise ValueError(f"no comma or end at character {at}")
            opened.pop()
            if holder.repeats:
                repeated.append((holder.start, holder.where))
            value, at = holder.container, at + 1
        else:
            return value, at, repeated


def _first(text: str, at: int, container: _Open) -> int:
    # An object's member begins with its name and a colon; an array's
    # element begins with its value (RFC 8259, 4. Objects).
    if container.close == "]":
        return at
    if text[at : at + 1] != '"':
        raise ValueError(f"no name at character {at}")
    container.name, at = _string(text, at)
    at = _whitespace(text, at)
    if text[at : at + 1] != ":":
        raise ValueError(f"no colon at character {at}")
    return _whitespace(text, at + 1)


def _scalar(text: str, at: int) -> tuple[Any, int]:
    # A value that holds none: a string, a number, or one of the three
    # literal names (RFC 8259, 3. Values).
    char = text[at : at + 1]
    if char == '"':
        return _string(text, at)
    if char and char in "-" + _DIGITS:
        return _number(text, at)
    for name, literal in _LITERALS.items():
        if text.startswith(name, at):
            return literal, at + len(name)
    raise ValueError(f"no JSON value at character {at}")


def _text(value: Any, indent: int) -> str:
    # A value is written left to right, what is still to write kept in a
    # list, so any depth of nesting is written: a text piece as it is,
    # a value with its depth (spec › value.3).
    parts: list[str] = []
    waiting: list[Any] = [(value, 1)]
    while waiting:
        item = waiting.pop()
        if isinstance(item, str):
            parts.append(item)
            continue
        value, depth = item
        found = primitive_type(value)
        if found not in ("object", "array") or not value:
            parts.append(_simple(value, found))
            continue
        inner = "\n" + " " * (indent * depth) if indent else ""
        outer = "\n" + " " * (indent * (depth - 1)) if indent else ""
        colon = ": " if indent else ":"
        if found == "object":
            children = [(encode_string(k) + colon, v) for k, v in value.items()]
            ends = "{}"
        else:
            children = [("", v) for v in value]
            ends = "[]"
        sequence: list[Any] = [ends[0]]
        for i, (prefix, child) in enumerate(children):
            sequence += [("," if i else "") + inner + prefix, (child, depth + 1)]
        sequence.append(outer + ends[1])
        waiting += reversed(sequence)
    return "".join(parts)


def _simple(value: Any, found: str) -> str:
    # An empty object or array, a string, a number, or a literal name.
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


def _string(text: str, start: int) -> tuple[str, int]:
    # A string is characters between quotation marks; the quotation
    # mark, the reverse solidus and the control characters are escaped,
    # and a character outside the Basic Multilingual Plane may be
    # escaped as its UTF-16 surrogate pair (RFC 8259, 7. Strings).
    chars = []
    at = start + 1
    while True:
        run = _UNESCAPED.match(text, at)
        if run:
            chars.append(run.group())
            at = run.end()
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
