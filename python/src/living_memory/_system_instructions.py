"""The system instructions schema.

OTEL-GENAI model/gen-ai/gen-ai-system-instructions.json.
"""

__all__ = ["rows", "sheets", "validates", "ATTRIBUTE"]

from typing import Any

from living_memory import _json_schema, _parts
from living_memory._relations import Key, Variants

ATTRIBUTE = "gen_ai.system_instructions"

# The schema is written out by hand, without its title and
# description annotations (OTEL-GENAI,
# model/gen-ai/gen-ai-system-instructions.json).
_SCHEMA: dict[str, Any] = {
    "$defs": _parts.definitions("GenericPart", "TextPart"),
    "items": {"anyOf": [{"$ref": "#/$defs/TextPart"}, {"$ref": "#/$defs/GenericPart"}]},
    "type": "array",
}

_PARTS = Variants(
    {"text": _parts.TEXT, "generic": _parts.GENERIC},
    _parts.belongs_to({"text": "TextPart"}, _SCHEMA),
)


def sheets() -> list[tuple[str, list[str]]]:
    return _PARTS.sheets(ATTRIBUTE)


def validates(value: Any) -> bool:
    return _json_schema.validates(value, _SCHEMA, _SCHEMA)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _PARTS.array_rows(ATTRIBUTE, Key((address, "")), value)
