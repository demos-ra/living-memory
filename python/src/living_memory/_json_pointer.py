"""How a JSON Pointer is written, and what it identifies in a value."""

__all__ = ["evaluate", "pointer"]

from typing import Any


def pointer(at: str, token: str | int) -> str:
    # '~' is written '~0' and '/' is written '~1' in a reference token
    # (RFC 6901, Section 3).
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{at}/{escaped}"


def evaluate(document: Any, at: str) -> Any:
    # Each token is unescaped '~1' first, then '~0', and names a member
    # of an object or the zero-based index of an array element
    # (RFC 6901, Section 4).
    found = document
    for token in at.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        found = found[int(token)] if isinstance(found, list) else found[token]
    return found
