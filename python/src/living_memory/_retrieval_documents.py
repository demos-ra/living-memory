"""The retrieval documents schema.

OTEL-GENAI model/gen-ai/gen-ai-retrieval-documents.json.
"""

__all__ = ["rows", "sheets", "validates", "ATTRIBUTE"]

from typing import Any

from living_memory import _json_schema
from living_memory._relations import Definition, Key

ATTRIBUTE = "gen_ai.retrieval.documents"
_DOCUMENT = Definition(
    columns=("id", "score"),
    lines=frozenset({"id"}),
    properties=frozenset({"id", "score"}),
)

# The schema is written out by hand, without its title and
# description annotations (OTEL-GENAI,
# model/gen-ai/gen-ai-retrieval-documents.json).
_SCHEMA: dict[str, Any] = {
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


def sheets() -> list[tuple[str, list[str]]]:
    return _DOCUMENT.sheets(ATTRIBUTE)


def validates(value: Any) -> bool:
    return _json_schema.validates(value, _SCHEMA, _SCHEMA)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _DOCUMENT.array_rows(ATTRIBUTE, Key((address, "")), value)
