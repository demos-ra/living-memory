"""The system instructions schema.

OTEL-GENAI model/gen-ai/gen-ai-system-instructions.json: an array of
parts, each a TextPart or a GenericPart.

Functions:
entries -- return the rows of a gen_ai.system_instructions value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
PARTS -- the parts, by type value
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "PARTS", "SHEETS"]

from typing import Any

from living_memory import _parts, _relations

ATTRIBUTE = "gen_ai.system_instructions"
PARTS = {"text": _parts.TEXT, "generic": _parts.GENERIC}
SHEETS = _relations.variant_sheets(ATTRIBUTE, PARTS)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.system_instructions value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.variant_entries(ATTRIBUTE, PARTS, address, "", value)
