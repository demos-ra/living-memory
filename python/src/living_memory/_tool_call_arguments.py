"""The tool call arguments schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-call-arguments.json: an object of
any properties, so a value of any shape.

Functions:
entries -- return the rows of a gen_ai.tool.call.arguments value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "SHEETS"]

from typing import Any

from living_memory import _relations

ATTRIBUTE = "gen_ai.tool.call.arguments"
SHEETS = _relations.node_sheets(ATTRIBUTE)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.tool.call.arguments value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.node_entries(ATTRIBUTE, [address], value, "")
