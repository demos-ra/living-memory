"""How text is written in MTSV fields, and what a field cannot hold."""

__all__ = [
    "HIGH_SURROGATES",
    "LOW_SURROGATES",
    "carried",
    "holds_separator",
    "name_carried",
    "runs",
    "text",
]

import re
from typing import Any

# The UTF-16 surrogates are high, then low; an unpaired one encodes no
# Unicode character, and so is not text (RFC 8259, 8.2. Unicode
# Characters; MTSV draft, Generators).
HIGH_SURROGATES = range(0xD800, 0xDC00)
LOW_SURROGATES = range(0xDC00, 0xE000)
_SURROGATE = f"[{chr(HIGH_SURROGATES.start)}-{chr(LOW_SURROGATES.stop - 1)}]"
# A CR not followed by LF is not written, nor is a character that is
# not text (MTSV draft, Generators; spec › field.4).
_NOT_TEXT = re.compile("\r(?!\n)|" + _SURROGATE)
# FF, a line break and HT separate sheets, records and fields (MTSV
# draft, Separators).
_SEPARATOR = re.compile("[\f\n\t]")
# A name within a pointer holds no HT, LF, FF or CR, nor what is not
# text (spec › field.4).
_NOT_IN_NAME = re.compile("[\t\n\f\r]|" + _SURROGATE)
_LINE_BREAK = re.compile("\r?\n")


def text(value: Any) -> str:
    # A string is its characters, a number its text as written, a
    # boolean true or false, and null, an object and an array an empty
    # field (RFC 8259, 3. Values; CSVW, 4.5 Cells; spec › field.2).
    if value is True:
        return "true"
    if value is False:
        return "false"
    return value if isinstance(value, str) else ""


def carried(value: str) -> tuple[str, bool]:
    # A value keeps what a field can hold, and says whether anything was
    # left out (spec › field.4).
    kept = _NOT_TEXT.sub("", value)
    return kept, kept != value


def name_carried(name: str) -> tuple[str, bool]:
    # A member name within a pointer also loses HT, LF, FF and CR
    # (spec › field.4).
    kept = _NOT_IN_NAME.sub("", name)
    return kept, kept != name


def holds_separator(value: str) -> bool:
    return _SEPARATOR.search(value) is not None


def runs(value: str) -> list[tuple[int, int, int, str]]:
    # A string is split at each FF into pages, each page at each line
    # break, LF or CRLF, into lines, and each line at each HT into runs,
    # each with its zero-based page, line and position (RFC 20, 5.2
    # Control Characters; spec › field.3).
    found = []
    for page, page_text in enumerate(value.split("\f")):
        for line, line_text in enumerate(_LINE_BREAK.split(page_text)):
            for position, run in enumerate(line_text.split("\t")):
                found.append((page, line, position, run))
    return found
