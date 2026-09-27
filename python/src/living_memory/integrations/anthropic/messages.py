"""What a model is given and generates, in the Messages API's schema."""

__all__ = ["definitions", "units"]

from importlib.resources import files
from typing import Any

from living_memory import _json

# The schema generated from the SDK's beta types, beside this module
# (messages.mtsv › schema.10).
_SCHEMA = "messages.schema.json"


def units() -> dict[str, Any]:
    """Return the schema of each kept unit, by the member that holds it.

    The members are system, a string system prompt or one of its blocks;
    tools, one tool; messages, one message; and content, a response's
    content (messages.mtsv › kept.1).
    """
    parts = _generated()["properties"]
    return {
        "system": {"anyOf": [_element(each) for each in parts["system"]["anyOf"]]},
        "tools": _element(parts["tools"]),
        "messages": _element(parts["messages"]),
        "content": parts["content"],
    }


def definitions() -> dict[str, Any]:
    """Return each named type of the kept parts, by its name.

    Each is a definition that the units reach by $ref (messages.mtsv ›
    schema.7).
    """
    return _generated()["definitions"]


def _generated() -> dict[str, Any]:
    return _json.decode(files(__package__).joinpath(_SCHEMA).read_bytes())


def _element(schema: dict[str, Any]) -> dict[str, Any]:
    # A list's unit is each of its elements.
    return schema["items"] if schema.get("type") == "array" else schema
