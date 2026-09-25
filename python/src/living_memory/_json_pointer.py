"""How a JSON Pointer is written and evaluated, and a place named."""

__all__ = ["PlacedError", "evaluate", "pointer"]

from typing import Any


class PlacedError(ValueError):
    # A JSON document that does not conform names the place where it
    # fails by the JSON Pointer to it (RFC 6901, 1. Introduction; spec ›
    # module.1, value.3).
    def __init__(self, msg: str, at: str) -> None:
        super().__init__(msg, at)
        self.msg = msg
        self.pointer = at


def pointer(at: str, token: str | int) -> str:
    # A reference token follows its pointer after '/', with '~' written
    # '~0' and '/' written '~1' (RFC 6901, 3. Syntax).
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{at}/{escaped}"


def evaluate(document: Any, at: str) -> Any:
    # Each token is unescaped, '~1' first and then '~0', and names a
    # member of an object or the zero-based index of an array element
    # (RFC 6901, 4. Evaluation).
    found = document
    for token in at.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        found = found[int(token)] if isinstance(found, list) else found[token]
    return found
