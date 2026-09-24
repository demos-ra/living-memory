"""The system instructions schema.

OTEL-GENAI model/gen-ai/gen-ai-system-instructions.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory import _parts
from living_memory._relations import Key, Variants

ATTRIBUTE = "gen_ai.system_instructions"

# OTEL-GENAI model/gen-ai/gen-ai-system-instructions.json, written out
# by hand without its title and description annotations.
SCHEMA: dict[str, Any] = {
    "$defs": {
        "GenericPart": _parts.DEFS["GenericPart"],
        "TextPart": _parts.DEFS["TextPart"],
    },
    "items": {
        "anyOf": [{"$ref": "#/$defs/TextPart"}, {"$ref": "#/$defs/GenericPart"}]
    },
    "type": "array",
}

_PARTS = Variants(
    {"text": _parts.TEXT, "generic": _parts.GENERIC},
    _parts.belongs_to({"text": "TextPart"}, SCHEMA),
)
SHEETS = _PARTS.sheets(ATTRIBUTE)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return _PARTS.array_rows(ATTRIBUTE, Key((address, "")), value)
