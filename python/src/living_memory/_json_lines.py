"""How a JSON Lines file is split into its lines."""

__all__ = ["lines"]

from typing import TypeVar

_Text = TypeVar("_Text", str, bytes)

# JSON Lines, 1. UTF-8 Encoding: a byte order mark is not included.
_SIGNATURE = "﻿"


def lines(document: _Text) -> list[_Text]:
    # A file that begins with a byte order mark is refused; the lines
    # are those between LF, a terminator after the last being optional,
    # and a CR before LF is white space the JSON value ignores
    # (JSON Lines, 1. UTF-8 Encoding; 3. Line Terminator is '\n').
    if isinstance(document, str):
        signature, terminator = _SIGNATURE, "\n"
    else:
        signature, terminator = _SIGNATURE.encode(), b"\n"
    if document.startswith(signature):
        raise ValueError("a byte order mark is not written")
    found = document.split(terminator)
    if found[-1] == document[:0]:
        found.pop()
    return found
