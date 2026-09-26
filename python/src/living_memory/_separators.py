"""Which characters separate MTSV text, and what a field cannot hold."""

__all__ = ["cannot_hold", "holds_separator", "lines", "pages", "runs"]

import re

# HT separates fields, LF or CRLF records, and FF sheets (MTSV draft,
# Separators).
_FF = "\f"
_HT = "\t"
_LINE_BREAK = re.compile("\r?\n")
# A field or a sheet name holding HT, LF, FF or CR cannot be represented
# (MTSV draft, Generators).
_CANNOT_HOLD = re.compile("[\t\n\f\r]")
_SEPARATOR = re.compile("[\f\n\t]")


def cannot_hold(text: str) -> bool:
    return _CANNOT_HOLD.search(text) is not None


def holds_separator(text: str) -> bool:
    return _SEPARATOR.search(text) is not None


def pages(text: str) -> list[str]:
    return text.split(_FF)


def lines(text: str) -> list[str]:
    return _LINE_BREAK.split(text)


def runs(text: str) -> list[str]:
    return text.split(_HT)
