"""Which attributes are the eight, and how one of them is read."""

__all__ = ["check", "is_set", "rows", "sheets"]

from typing import Any

from living_memory import (
    _any_value,
    _input_messages,
    _memory_records,
    _output_messages,
    _retrieval_documents,
    _system_instructions,
    _tool_call_arguments,
    _tool_call_result,
    _tool_definitions,
)
from living_memory._json import decode

# The eight are found by their attribute keys, in the order of the
# Sheets (spec › set.1, file.4).
_SETS = {
    module.ATTRIBUTE: module
    for module in (
        _system_instructions,
        _tool_definitions,
        _input_messages,
        _output_messages,
        _tool_call_arguments,
        _tool_call_result,
        _memory_records,
        _retrieval_documents,
    )
}


def is_set(key: Any) -> bool:
    return key in _SETS


def sheets() -> list[tuple[str, list[str]]]:
    return [sheet for module in _SETS.values() for sheet in module.sheets()]


def rows(key: str, address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _SETS[key].rows(address, _value(value))


def check(key: str, value: Any, at: str) -> None:
    # A value of the eight that is not a JSON value, or that does not
    # validate against its schema, is non-conforming (spec › set.3).
    try:
        instance = _value(value)
    except ValueError as error:
        message = f"{at}: the JSON string of {key} is not JSON: {error}"
        raise ValueError(message) from error
    if not _SETS[key].validates(instance):
        raise ValueError(f"{at}: {key} does not validate against its schema")


def _value(value: Any) -> Any:
    # One of the eight is structured, or, on a span, a JSON string
    # (spec › set.1).
    name, member = _any_value.winner(value)
    if name == "stringValue":
        return decode(member)
    return _any_value.represent(value)
