"""The tool definitions schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-definitions.json: an array of
tools, each a FunctionToolDefinition or a GenericToolDefinition.

Functions:
entries -- return the rows of a gen_ai.tool.definitions value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
FUNCTION -- FunctionToolDefinition
GENERIC -- GenericToolDefinition
TOOLS -- the tools, by type value
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "FUNCTION", "GENERIC", "TOOLS", "SHEETS"]

from typing import Any

from living_memory import _relations
from living_memory._relations import Shape

ATTRIBUTE = "gen_ai.tool.definitions"
FUNCTION = Shape(
    columns=("name", "description"),
    lines=frozenset({"name", "description"}),
    nodes=("parameters",),
    known=frozenset({"type", "name", "description", "parameters"}),
)
GENERIC = Shape(
    columns=("type", "name"),
    lines=frozenset({"type", "name"}),
    known=frozenset({"type", "name"}),
)
TOOLS = {"function": FUNCTION, "generic": GENERIC}
SHEETS = _relations.variant_sheets(ATTRIBUTE, TOOLS)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.tool.definitions value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.variant_entries(ATTRIBUTE, TOOLS, address, "", value)
