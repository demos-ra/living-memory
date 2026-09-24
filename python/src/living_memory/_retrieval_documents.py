"""The retrieval documents schema.

OTEL-GENAI model/gen-ai/gen-ai-retrieval-documents.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory._relations import Definition, Key

ATTRIBUTE = "gen_ai.retrieval.documents"
_DOCUMENT = Definition(
    columns=("id", "score"),
    lines=frozenset({"id"}),
    properties=frozenset({"id", "score"}),
)
SHEETS = _DOCUMENT.sheets(ATTRIBUTE)

# OTEL-GENAI model/gen-ai/gen-ai-retrieval-documents.json, written out
# by hand without its title and description annotations.
SCHEMA: dict[str, Any] = {
    "$defs": {
        "RetrievalDocument": {
            "additionalProperties": True,
            "properties": {
                "id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
                "score": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "default": None,
                },
            },
            "type": "object",
        }
    },
    "items": {"$ref": "#/$defs/RetrievalDocument"},
    "type": "array",
}

def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _DOCUMENT.array_rows(ATTRIBUTE, Key((address, "")), value)
