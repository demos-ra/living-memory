"""The output messages schema.

OTEL-GENAI model/gen-ai/gen-ai-output-messages.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory import _parts
from living_memory._relations import Definition, Key

ATTRIBUTE = "gen_ai.output.messages"
_MESSAGE = Definition(
    columns=("role", "name", "finish_reason"),
    lines=frozenset({"role", "name", "finish_reason"}),
    variants=(("parts", _parts.MESSAGE_PARTS),),
    properties=frozenset({"role", "parts", "name", "finish_reason"}),
)
SHEETS = _MESSAGE.sheets(ATTRIBUTE)

# OTEL-GENAI model/gen-ai/gen-ai-output-messages.json, written out by
# hand without its title and description annotations.
SCHEMA: dict[str, Any] = {
    "$defs": {
        **_parts.DEFS,
        "FinishReason": {
            "enum": [
                "stop",
                "length",
                "content_filter",
                "tool_call",
                "compaction",
                "error",
            ],
            "type": "string",
        },
        "OutputMessage": {
            "additionalProperties": True,
            "properties": {
                "role": {"anyOf": [{"$ref": "#/$defs/Role"}, {"type": "string"}]},
                "parts": {"items": _parts.ITEMS, "type": "array"},
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
                "finish_reason": {
                    "anyOf": [
                        {"$ref": "#/$defs/FinishReason"},
                        {"type": "string"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "deprecated": True,
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
    "items": {"$ref": "#/$defs/OutputMessage"},
    "type": "array",
}

def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _MESSAGE.array_rows(ATTRIBUTE, Key((address, "")), value)
