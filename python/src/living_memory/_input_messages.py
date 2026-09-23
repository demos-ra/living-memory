"""The input messages schema.

OTEL-GENAI model/gen-ai/gen-ai-input-messages.json: an array of
ChatMessage, each with its parts.

Functions:
entries -- return the rows of a gen_ai.input.messages value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
MESSAGE -- ChatMessage
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "MESSAGE", "SHEETS"]

from typing import Any

from living_memory import _parts, _relations
from living_memory._relations import Shape

ATTRIBUTE = "gen_ai.input.messages"
MESSAGE = Shape(
    columns=("role", "name"),
    lines=frozenset({"role", "name"}),
    variants=(("parts", _parts.MESSAGE_PARTS),),
    known=frozenset({"role", "parts", "name"}),
)
SHEETS = _relations.shape_sheets(ATTRIBUTE, MESSAGE)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.input.messages value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.array_entries(ATTRIBUTE, MESSAGE, address, "", value)
