"""How an OTLP JSON Lines file is read.

Its lines, their order and index, and UTF-8.

Functions:
load -- read MTSV sheets from a binary OTLP JSON Lines file
loads -- read MTSV sheets from an OTLP JSON Lines string
"""

__all__ = ["load", "loads"]

from typing import Any, BinaryIO

from living_memory import _relations, _telemetry_data


def load(fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a binary OTLP JSON Lines file.

    fp -- a binary file object open for reading, by position only

    spec › file.1: UTF-8; an invalid UTF-8 sequence is read as U+FFFD.
    Return the sheets. Raise ValueError for a line that is not JSON.
    """
    return loads(fp.read().decode("utf-8", errors="replace"))


def loads(s: str, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from an OTLP JSON Lines string.

    s -- the text of the file, by position only

    spec › file.2: the lines are read in the file's order, and nothing
    is sorted. Return every sheet of the output. Raise ValueError for a
    line that is not JSON, naming it by its number.
    """
    found: list[tuple[str, list[str]]] = []
    for index, line in enumerate(_lines(s)):
        try:
            data = _relations.decode(line)
        except ValueError as error:
            # JSON Lines, Conventions: the first value is "value 1".
            raise ValueError(f"line {index + 1}: {error}") from None
        found += _telemetry_data.entries(index, data)
    return _relations.assemble(_telemetry_data.SHEETS, found)


def _lines(s: str) -> list[str]:
    """Return the lines of a JSON Lines text.

    s -- the text

    OTEL-FILE-EXPORTER, JSON lines file: one JSON value per line, LF
    between lines. JSON Lines, Line Terminator is '\\n': a line
    terminator after the last value is the last byte in the file.
    """
    found = s.split("\n")
    if found[-1] == "":
        found.pop()
    return found
