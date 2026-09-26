"""The converter: the input and its module specification, as MTSV."""

__all__ = ["NonConformingError", "NonConformingInputError", "convert", "sheets"]

import logging
from collections.abc import Iterable
from typing import Any

from living_memory import _file, _json, _module, _relation, _value
from living_memory._json_pointer import PlacedError

# A library names its logger after its module and attaches no handler
# (Logging HOWTO, Configuring Logging for a Library).
_logger = logging.getLogger(__name__)


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


def convert(values: Iterable[bytes], schema: bytes, start: int = 0) -> str:
    """Convert JSON values that a schema describes into one MTSV file.

    values -- the input values, each a JSON text in UTF-8, in order
    schema -- the JSON Schema, a JSON text
    start -- the position of the first value in the whole input, for
        an input converted in parts

    Return the MTSV text. Raise NonConformingError for a module
    specification that does not conform, and NonConformingInputError
    for an input value. What a field cannot hold is logged as a
    warning, by its pointer.
    """
    root, sheet_layout = _read(schema)
    placed: list[_relation.Placed] = []
    reports: list[str] = []
    for position, data in enumerate(values, start):
        try:
            value = _value.read(data, root)
        except PlacedError as error:
            raise NonConformingInputError(error.msg, position, error.pointer) from None
        found, missed = _relation.place(sheet_layout, value, position)
        placed += found
        reports += missed
    # What a field cannot hold is reported as a warning, by its pointer
    # as a JSON string (Logging HOWTO, When to use logging; spec ›
    # field.3).
    for at in reports:
        _logger.warning("not carried: %s", _json.encode_string(at))
    return _file.write(sheet_layout, placed)


def sheets(schema: bytes) -> list[str]:
    """Return the names of a schema's sheets, in the file's order.

    A file holds only the sheets that hold a record, so the parts of an
    input converted in parts are appended sheet by sheet in this order.
    Raise NonConformingError for a module specification that does not
    conform.
    """
    return _file.names(_read(schema)[1])


def _read(schema: bytes) -> tuple[Any, _relation.Layout]:
    # A module specification that does not conform is rejected, naming
    # the place in its schema (spec › module.1, module.2).
    try:
        root = _module.read(schema)
    except PlacedError as error:
        raise NonConformingError(f"the schema: {error.msg}", error.pointer) from None
    return root, _relation.layout(root)
