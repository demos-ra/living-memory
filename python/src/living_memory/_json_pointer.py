"""How a JSON Pointer is written, read from a URI fragment and
evaluated, and a place named."""

__all__ = ["PlacedError", "evaluate", "from_fragment", "pointer", "tokens"]

from typing import Any

from living_memory import _utf_8

# An array index is "0", or digits without a leading "0" (RFC 6901, 4.
# Evaluation).
_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"
# A URI fragment holds pchar, '/' and '?', and any other character
# percent-encoded: pchar is the unreserved characters, the
# sub-delimiters, ':' and '@' (RFC 3986, 3.5. Fragment; 3.3. Path; 2.3.
# Unreserved Characters; 2.2. Reserved Characters).
_FRAGMENT = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "-._~!$&'()*+,;=:@/?"
)


class PlacedError(ValueError):
    # A JSON document that does not conform names the place where it
    # fails by the JSON Pointer to it (RFC 6901, 1. Introduction; spec ›
    # schema.9, value.10).
    def __init__(self, msg: str, at: str) -> None:
        super().__init__(msg, at)
        self.msg = msg
        self.pointer = at


def pointer(at: str, token: str | int) -> str:
    # A reference token follows its pointer after '/', with '~' written
    # '~0' and '/' written '~1' (RFC 6901, 3. Syntax; spec › key.4).
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{at}/{escaped}"


def tokens(at: str) -> list[str]:
    # Each token is unescaped, '~1' first and then '~0' (RFC 6901, 4.
    # Evaluation).
    return [token.replace("~1", "/").replace("~0", "~") for token in at.split("/")[1:]]


def evaluate(document: Any, at: str) -> Any:
    # Each token names a member of an object, or an array element by its
    # zero-based index, which has no leading zero; any other token fails
    # (RFC 6901, 4. Evaluation; spec › schema.5, schema.11).
    found = document
    for token in tokens(at):
        if isinstance(found, list):
            if not _is_index(token):
                raise ValueError(f"{token!r} is not an array index")
            found = found[int(token)]
        else:
            found = found[token]
    return found


def from_fragment(fragment: str) -> str:
    # A JSON Pointer in its URI fragment representation is encoded in
    # UTF-8, the characters a fragment does not allow percent-encoded;
    # decoded, it is empty or each token follows '/' (RFC 6901, 3.
    # Syntax; 6. URI Fragment Identifier Representation; spec ›
    # schema.6, schema.11).
    data = bytearray()
    at = 0
    while at < len(fragment):
        char = fragment[at]
        if char == "%":
            digits = fragment[at + 1 : at + 3]
            if len(digits) < 2 or not all(d in _HEX_DIGITS for d in digits):
                raise ValueError(f"{digits!r} is no octet")
            data.append(int(digits, 16))
            at += 3
            continue
        if char not in _FRAGMENT:
            raise ValueError(f"{char!r} is not a fragment's")
        data += char.encode("ascii")
        at += 1
    try:
        found = _utf_8.decode(bytes(data))
    except ValueError:
        raise ValueError("its octets are not UTF-8") from None
    if found and not found.startswith("/"):
        raise ValueError("its fragment is not a JSON Pointer")
    return found


def _is_index(token: str) -> bool:
    digits = bool(token) and all(char in _DIGITS for char in token)
    return digits and (token == "0" or token[0] != "0")
