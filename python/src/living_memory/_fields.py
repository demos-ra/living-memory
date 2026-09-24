"""How a value is written as the text of an MTSV field."""

__all__ = ["field", "lines", "text"]

from typing import Any

# The draft, Generators: a field holding HT, LF, FF or CR cannot be
# represented.
_SEPARATORS = frozenset("\t\n\f\r")


def text(value: Any) -> str:
    # A field is text (the draft, Data Model): a string is its own text,
    # a boolean true or false (RFC 8259, Section 3), and anything else,
    # null and absent included, the empty field that reads as null
    # (CSVW 4.5; spec › node.1, text.1).
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else ""


def lines(value: Any) -> list[str]:
    # A text holding a line break is carried as its lines, a CRLF read
    # as LF; a text with none has no lines (spec › line.1).
    written = text(value)
    if "\n" not in written:
        return []
    found = written.split("\n")
    return [line.removesuffix("\r") for line in found[:-1]] + found[-1:]


def field(value: str) -> str:
    # A generator does not write what a field cannot hold, so such a
    # field is left empty (the draft, Generators; spec › character.1).
    return "" if not _SEPARATORS.isdisjoint(value) else value
