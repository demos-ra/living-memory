"""The input messages schema.

OTEL-GENAI model/gen-ai/gen-ai-input-messages.json.
"""

__all__ = ["rows", "sheets", "validates", "ATTRIBUTE"]

from typing import Any

from living_memory import _json_schema, _parts
from living_memory._relations import Definition, Key

ATTRIBUTE = "gen_ai.input.messages"
_MESSAGE = Definition(
    columns=("role", "name"),
    lines=frozenset({"role", "name"}),
    variants=(("parts", _parts.MESSAGE_PARTS),),
    properties=frozenset({"role", "parts", "name"}),
)

# The schema is written out by hand, without its title and
# description annotations (OTEL-GENAI,
# model/gen-ai/gen-ai-input-messages.json).
_SCHEMA: dict[str, Any] = {
    "$defs": {
        **_parts.definitions(),
        "ChatMessage": {
            "additionalProperties": True,
            "properties": {
                "role": {"anyOf": [{"$ref": "#/$defs/Role"}, {"type": "string"}]},
                "parts": {"items": _parts.items(), "type": "array"},
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["role", "parts"],
            "type": "object",
        },
        "Role": {
            "enum": ["system", "user", "assistant", "tool"],
            "type": "string",
        },
    },
    "items": {"$ref": "#/$defs/ChatMessage"},
    "type": "array",
}


def sheets() -> list[tuple[str, list[str]]]:
    return _MESSAGE.sheets(ATTRIBUTE)


def validates(value: Any) -> bool:
    return _json_schema.validates(value, _SCHEMA, _SCHEMA)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _MESSAGE.array_rows(ATTRIBUTE, Key((address, "")), value)
