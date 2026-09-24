"""The tool definitions schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-definitions.json.
"""

__all__ = ["rows", "sheets", "validates", "ATTRIBUTE"]

from typing import Any

from living_memory import _json_schema, _parts
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
# The schema is written out by hand, without its title and
# description annotations (OTEL-GENAI,
# model/gen-ai/gen-ai-tool-definitions.json).
_SCHEMA: dict[str, Any] = {
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
    _parts.belongs_to({"function": "FunctionToolDefinition"}, _SCHEMA),
)


def sheets() -> list[tuple[str, list[str]]]:
    return _TOOLS.sheets(ATTRIBUTE)


def validates(value: Any) -> bool:
    return _json_schema.validates(value, _SCHEMA, _SCHEMA)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _TOOLS.array_rows(ATTRIBUTE, Key((address, "")), value)
