"""How an OTLP JSON Lines file is read."""

__all__ = ["OTLPDecodeError", "load", "loads"]

import logging
from typing import Any, BinaryIO

from living_memory import _relations, _telemetry_data
from living_memory._json import decode

_logger = logging.getLogger("living_memory.integrations")

# JSON Lines, 1. UTF-8 Encoding.
_SIGNATURE = chr(0xFEFF)


class OTLPDecodeError(ValueError):
    """Subclass of ValueError with the following additional properties:

    msg: The unformatted error message
    lineno: The line of the file, the first being line 1
    """

    def __init__(self, msg: str, lineno: int) -> None:
        super().__init__(f"{msg}: line {lineno}")
        self.msg = msg
        self.lineno = lineno

    def __reduce__(self) -> tuple[type, tuple[str, int]]:
        return self.__class__, (self.msg, self.lineno)


def load(fp: BinaryIO, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a binary OTLP JSON Lines file.

    An invalid UTF-8 sequence is read as U+FFFD. Raise TypeError for a
    file opened in text mode, and OTLPDecodeError for a file that is not
    OTLP JSON Lines.
    """
    # spec › file.1.
    b = fp.read()
    try:
        s = b.decode("utf-8", errors="replace")
    except AttributeError:
        raise TypeError(
            "File must be opened in binary mode, e.g. use `open('foo.jsonl', 'rb')`"
        ) from None
    return loads(s)


def loads(s: str, /) -> list[dict[str, Any]]:
    """Read MTSV sheets from an OTLP JSON Lines string.

    What the input holds but the sheets do not carry is logged as a
    warning, its record's left_behind attribute holding the sorted
    names. Raise TypeError for anything but a string, and
    OTLPDecodeError for a file that is not OTLP JSON Lines.
    """
    # spec › file.2, file.5, set.3.
    if not isinstance(s, str):
        raise TypeError(f"Expected str object, not '{type(s).__qualname__}'")
    if s.startswith(_SIGNATURE):
        raise OTLPDecodeError("a byte order mark is not written", 1)
    found: list[tuple[str, list[str]]] = []
    left: set[str] = set()
    kind = ""
    for index, line in enumerate(_lines(s)):
        try:
            data = decode(line)
            _telemetry_data.check(data)
        except ValueError as error:
            raise OTLPDecodeError(str(error), index + 1) from None
        this = _telemetry_data.kind_of_data(data)
        if this and kind and this != kind:
            message = f"a file holds one kind of data, not {kind} and {this}"
            raise OTLPDecodeError(message, index + 1)
        kind = kind or this
        left |= _telemetry_data.left_behind(data)
        found += _telemetry_data.rows(index, data)
    sheets, emptied = _relations.assemble(_telemetry_data.SHEETS, found)
    _report(left | emptied)
    return sheets


def _lines(s: str) -> list[str]:
    # OTEL-FILE-EXPORTER, JSON lines file; JSON Lines, 3. Line
    # Terminator is '\n'.
    found = s.split("\n")
    if found[-1] == "":
        found.pop()
    return found


def _report(names: set[str]) -> None:
    # Logging HOWTO, When to use logging; logging, Logger.debug.
    if not names:
        return
    ordered = sorted(names)
    _logger.warning(
        "left behind: %s", ", ".join(ordered), extra={"left_behind": ordered}
    )
