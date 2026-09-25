"""How the input becomes files."""

__all__ = ["NonConformingError", "NonConformingInputError", "convert"]

import logging
from collections.abc import Iterable
from typing import Any

import mtsv

from living_memory import _fields, _json, _json_pointer, _json_schema, _relations
from living_memory._json_pointer import PlacedError

# A library names its logger after its module and attaches no handler
# (Logging HOWTO, Configuring Logging for a Library).
_logger = logging.getLogger(__name__)

# MTSV files take the extension .mtsv (MTSV draft, Media Type
# Registration; spec › file.1).
_EXTENSION = ".mtsv"


class NonConformingError(ValueError):
    """A module specification that does not conform.

    Subclass of ValueError with the following additional properties:

    msg: The unformatted error message
    pointer: The JSON Pointer to the place within the schema
    """

    def __init__(self, msg: str, pointer: str) -> None:
        super().__init__(msg, pointer)
        self.msg = msg
        self.pointer = pointer

    def __str__(self) -> str:
        where = _json.encode_string(self.pointer)
        return f"{self.msg}: the module specification, {where}"

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return self.__class__, (self.msg, self.pointer)


class NonConformingInputError(NonConformingError):
    """An input value that does not conform.

    Subclass of NonConformingError with the following additional
    property:

    position: The input value's position, counted from 0; pointer is
        the place within the value
    """

    def __init__(self, msg: str, position: int, pointer: str) -> None:
        super().__init__(msg, pointer)
        self.position = position

    def __str__(self) -> str:
        where = _json.encode_string(self.pointer)
        return f"{self.msg}: value {self.position}, {where}"

    def __reduce__(self) -> tuple[type, tuple[str, int, str]]:
        return self.__class__, (self.msg, self.position, self.pointer)


def convert(
    values: Iterable[bytes], schema: bytes, pointer: bytes = b""
) -> dict[str, str]:
    """Convert JSON values that a schema describes into MTSV files.

    values -- the input values, each a JSON text in UTF-8, in order
    schema -- the JSON Schema, a JSON text
    pointer -- the JSON Pointer, as a JSON string, of the member whose
        value selects each value's file, or empty for one file

    Return a dict of each file's name and its MTSV text. Without a
    pointer, the one file is named "" for the caller to name. Raise
    NonConformingError for a module specification that does not
    conform, and NonConformingInputError for an input value. What a
    field cannot hold is logged as a warning, by its pointer.
    """
    root = _module(schema)
    member = _file_member(pointer) if pointer else ""
    sheet_layout = _layout(root)
    files: dict[str, list[Any]] = {} if pointer else {"": []}
    reports: list[str] = []
    for position, data in enumerate(values):
        value = _value(data, position, root)
        name = _file_name(value, member, position) if pointer else ""
        found, missed = _relations.records(sheet_layout, value, position)
        files[name] = files.get(name, []) + found
        reports += missed
    for at in reports:
        _logger.warning("not carried: %s", _json.encode_string(at))
    return {
        name: mtsv.dumps(_relations.sheets(sheet_layout, found))
        for name, found in files.items()
    }


def _module(schema: bytes) -> Any:
    # The schema is a JSON text whose names are unique, whose every
    # $ref resolves within it and whose every pattern is of the subset
    # (spec › module.1, module.3).
    try:
        root = _json.decode(schema)
        _json_schema.check(root, root)
    except PlacedError as error:
        raise NonConformingError(f"the schema: {error.msg}", error.pointer) from None
    return root


def _file_member(pointer: bytes) -> str:
    # The member that selects the file is a JSON Pointer as a JSON
    # string (spec › module.2).
    # A pointer that is not one fails whole, and so is named by "".
    try:
        member = _json.decode(pointer)
        if _json.type(member) == "string":
            return member
    except PlacedError:
        pass
    raise NonConformingError("the file pointer is not a JSON string", "")


def _layout(root: Any) -> _relations.Layout:
    # Every name the schema gives a sheet or a column is text a field
    # can hold (spec › module.1).
    try:
        return _relations.layout(root)
    except PlacedError as error:
        raise NonConformingError(f"the schema: {error.msg}", error.pointer) from None


def _value(data: bytes, position: int, root: Any) -> Any:
    # A value is a JSON text whose names are unique and that validates;
    # the place named is the deepest where an assertion fails (spec ›
    # value.1, value.3).
    try:
        value = _json.decode(data)
    except PlacedError as error:
        raise NonConformingInputError(error.msg, position, error.pointer) from None
    if not _json_schema.validates(value, root, root):
        place = _json_schema.locate(value, root, root)
        raise NonConformingInputError("an assertion fails", position, place)
    return value


def _file_name(value: Any, member: str, position: int) -> str:
    # A file is named by the member's text and .mtsv; a value without
    # the member is non-conforming (spec › file.1, value.3).
    try:
        selecting = _json_pointer.evaluate(value, member)
    except (KeyError, IndexError, TypeError, ValueError):
        raise NonConformingInputError(
            "the file member is missing", position, member
        ) from None
    return _fields.text(selecting) + _EXTENSION
