"""How a value is written as a field's text, and what is not carried."""

__all__ = ["carried", "name_carried", "text", "type"]

import re
from typing import Any

from living_memory import _json, _separators

_SURROGATE = (
    f"[{chr(_json.HIGH_SURROGATES.start)}-{chr(_json.LOW_SURROGATES.stop - 1)}]"
)
# A CR not followed by LF is not written, nor is a character that is
# not text (RFC 8259, 8.2. Unicode Characters; MTSV draft, Generators;
# spec › field.3).
_NOT_TEXT = re.compile("\r(?!\n)|" + _SURROGATE)
# A member name within a pointer holds no HT, LF, FF or CR either (spec
# › field.3).
_NOT_IN_NAME = re.compile("[\t\n\f\r]|" + _SURROGATE)


def text(value: Any) -> str:
    # A string is its characters, a number its text as written, a
    # boolean true or false, and null, an object and an array an empty
    # field (RFC 8259, 3. Values; CSVW, 4.5 Cells; spec › field.1); a
    # string holding FF, a line break or HT is an empty field, its
    # text written as runs (spec › field.2).
    if value is True:
        return "true"
    if value is False:
        return "false"
    if not isinstance(value, str) or _separators.holds_separator(value):
        return ""
    return value


def type(value: Any) -> str:
    # In a sheet of instances, a record's type is its instance's
    # primitive type (JSON Schema, 4.2.1. Instance Data Model; spec ›
    # field.1).
    return _json.type(value)


def carried(value: str) -> tuple[str, bool]:
    # A string keeps the rest of its text, and says whether anything was
    # left out (spec › field.3).
    kept = _NOT_TEXT.sub("", value)
    return kept, kept != value


def name_carried(name: str) -> tuple[str, bool]:
    # A member name within a pointer keeps the rest of its text, and
    # says whether anything was left out (spec › field.3).
    kept = _NOT_IN_NAME.sub("", name)
    return kept, kept != name
