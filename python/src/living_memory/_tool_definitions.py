"""The tool definitions schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-definitions.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory import _parts
from living_memory._relations import Definition, Key, Variants

ATTRIBUTE = "gen_ai.tool.definitions"
_FUNCTION = Definition(
    columns=("name", "description"),
    lines=frozenset({"name", "description"}),
    nodes=("parameters",),
    properties=frozenset({"type", "name", "description", "parameters"}),
)
_GENERIC = Definition(
    columns=("type", "name"),
    lines=frozenset({"type", "name"}),
    properties=frozenset({"type", "name"}),
)
# OTEL-GENAI model/gen-ai/gen-ai-tool-definitions.json, written out by
# hand without its title and description annotations.
SCHEMA: dict[str, Any] = {
    "$defs": {
        "FunctionToolDefinition": {
            "additionalProperties": True,
            "properties": {
                "type": {"const": "function", "type": "string"},
                "name": {"type": "string"},
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
                "parameters": {
                    "anyOf": [
                        {"$ref": "http://json-schema.org/draft-07/schema#"},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["type", "name"],
            "type": "object",
        },
        "GenericToolDefinition": {
            "additionalProperties": True,
            "properties": {"type": {"type": "string"}, "name": {"type": "string"}},
            "required": ["type", "name"],
            "type": "object",
        },
    },
    "items": {
        "anyOf": [
            {"$ref": "#/$defs/FunctionToolDefinition"},
            {"$ref": "#/$defs/GenericToolDefinition"},
        ]
    },
    "type": "array",
}

_TOOLS = Variants(
    {"function": _FUNCTION, "generic": _GENERIC},
    _parts.belongs_to({"function": "FunctionToolDefinition"}, SCHEMA),
)
SHEETS = _TOOLS.sheets(ATTRIBUTE)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _TOOLS.array_rows(ATTRIBUTE, Key((address, "")), value)
