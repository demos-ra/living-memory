"""The memory records schema.

OTEL-GENAI model/gen-ai/gen-ai-memory-records.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory._relations import Definition, Key

ATTRIBUTE = "gen_ai.memory.records"
_RECORD = Definition(
    columns=("id", "score"),
    lines=frozenset({"id"}),
    nodes=("content", "metadata"),
    properties=frozenset({"content", "id", "metadata", "score"}),
)
SHEETS = _RECORD.sheets(ATTRIBUTE)

# OTEL-GENAI model/gen-ai/gen-ai-memory-records.json, written out by
# hand without its title and description annotations.
SCHEMA: dict[str, Any] = {
    "$defs": {
        "MemoryRecord": {
            "additionalProperties": True,
            "properties": {
                "content": {},
                "id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
                "metadata": {
                    "anyOf": [
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                },
                "score": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["content"],
            "type": "object",
        }
    },
    "items": {"$ref": "#/$defs/MemoryRecord"},
    "type": "array",
}

def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _RECORD.array_rows(ATTRIBUTE, Key((address, "")), value)
