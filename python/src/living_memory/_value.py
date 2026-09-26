"""How each input value is read, and when it is rejected."""

__all__ = ["read"]

from typing import Any

from living_memory import _json, _json_schema
from living_memory._json_pointer import PlacedError


def read(data: bytes, root: Any) -> Any:
    # A value is a JSON text in UTF-8, its members in the order written
    # and its numbers as written, whose names are unique and that
    # validates against the schema; the place named is the deepest
    # instance location where an assertion fails (spec › value.1-3).
    value = _json.decode(data)
    if not _json_schema.validates(value, root, root):
        place = _json_schema.locate(value, root, root)
        raise PlacedError("an assertion fails", place)
    return value
