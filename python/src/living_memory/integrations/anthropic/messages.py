"""What a model is given and generates, in the Messages API's schema."""

__all__ = ["definitions", "ending", "kept", "units"]

from importlib.resources import files
from typing import Any

from living_memory import _json

# The schema generated from the SDK's beta types, beside this module
# (messages.mtsv › schema.10).
_SCHEMA = "messages.schema.json"
# The response's members kept beside its content: how it ended, and what
# the API changed in the input before showing it to the model
# (messages.mtsv › kept.1).
_ENDING = (
    "stop_reason",
    "stop_sequence",
    "stop_details",
    "input_transformations",
    "context_management",
)
# A cache control breakpoint, an instruction about delivery
# (messages.mtsv › kept.2).
_CONTROL = "cache_control"


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


def ending() -> dict[str, Any]:
    """Return the schema of each response member kept beside content.

    The members are stop_reason, stop_sequence and stop_details, how the
    response ended, and input_transformations and context_management,
    what the API changed in the input before showing it to the model
    (messages.mtsv › kept.1).
    """
    parts = _generated()["properties"]
    return {name: parts[name] for name in _ENDING}


def kept(member: str, unit: Any) -> Any:
    """Return a unit as it is kept, its cache control breakpoints out.

    member -- the member that holds the unit: system, tools, messages or
        content
    unit -- a system block or string, a tool, a message, or a response's
        content

    The breakpoint is left out of each system block, tool and content
    block, and of each block within a block's content or its source's
    content (messages.mtsv › kept.2).
    """
    if member == "messages" and isinstance(unit.get("content"), list):
        return {**unit, "content": [_block(each) for each in unit["content"]]}
    if member == "content":
        return [_block(each) for each in unit]
    return _block(unit)


def _block(block: Any) -> Any:
    # A block without its breakpoint, and the blocks within its content
    # and its source's content likewise; the blocks still to clear are
    # kept in a list, so blocks of any depth of nesting are cleared
    # (spec › value.3).
    top = _cleared(block)
    waiting = [top]
    while waiting:
        found = waiting.pop()
        if not isinstance(found, dict):
            continue
        if isinstance(found.get("content"), list):
            found["content"] = [_cleared(each) for each in found["content"]]
            waiting += found["content"]
        source = found.get("source")
        if isinstance(source, dict) and isinstance(source.get("content"), list):
            inner = [_cleared(each) for each in source["content"]]
            found["source"] = {**source, "content": inner}
            waiting += inner
    return top


def _cleared(block: Any) -> Any:
    # A block without its breakpoint, as a new object.
    if not isinstance(block, dict):
        return block
    return {k: v for k, v in block.items() if k != _CONTROL}


def _generated() -> dict[str, Any]:
    return _json.decode(files(__package__).joinpath(_SCHEMA).read_bytes())


def _element(schema: dict[str, Any]) -> dict[str, Any]:
    # A list's unit is each of its elements.
    return schema["items"] if schema.get("type") == "array" else schema
