"""The output messages schema.

OTEL-GENAI model/gen-ai/gen-ai-output-messages.json: an array of
OutputMessage, each with its parts.

Functions:
entries -- return the rows of a gen_ai.output.messages value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
MESSAGE -- OutputMessage
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "MESSAGE", "SHEETS"]

from typing import Any

from living_memory import _parts, _relations
from living_memory._relations import Shape

ATTRIBUTE = "gen_ai.output.messages"
MESSAGE = Shape(
    columns=("role", "name", "finish_reason"),
    lines=frozenset({"role", "name", "finish_reason"}),
    variants=(("parts", _parts.MESSAGE_PARTS),),
    known=frozenset({"role", "parts", "name", "finish_reason"}),
)
SHEETS = _relations.shape_sheets(ATTRIBUTE, MESSAGE)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.output.messages value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.array_entries(ATTRIBUTE, MESSAGE, address, "", value)
