"""What a model is given and generates, in the Messages API's schema."""

__all__ = ["definitions", "units"]

from importlib.resources import files
from typing import Any

from living_memory import _json

# The schema generated from the SDK's beta types, beside this module
# (messages.mtsv › schema.4).
_SCHEMA = "messages.schema.json"


def units() -> dict[str, Any]:
    # The schema of each kept unit, by the member that holds it: a
    # system prompt that is a string, or one of its blocks; a tool; a
    # message; and a response's content (messages.mtsv › kept.1).
    parts = _generated()["properties"]
    return {
        "system": {"anyOf": [_element(each) for each in parts["system"]["anyOf"]]},
        "tools": _element(parts["tools"]),
        "messages": _element(parts["messages"]),
        "content": parts["content"],
    }


def definitions() -> dict[str, Any]:
    # Each named type of the kept parts, by its name (messages.mtsv ›
    # schema.3).
    return _generated()["definitions"]


def _generated() -> dict[str, Any]:
    return _json.decode(files(__package__).joinpath(_SCHEMA).read_bytes())


def _element(schema: dict[str, Any]) -> dict[str, Any]:
    # A list's unit is each of its elements.
    return schema["items"] if schema.get("type") == "array" else schema
