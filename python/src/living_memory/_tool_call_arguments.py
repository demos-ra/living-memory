"""The tool call arguments schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-call-arguments.json.
"""

__all__ = ["rows", "ATTRIBUTE", "SHEETS", "SCHEMA"]

from typing import Any

from living_memory._relations import Key, node_rows, node_sheets

ATTRIBUTE = "gen_ai.tool.call.arguments"
SHEETS = node_sheets(ATTRIBUTE)

# OTEL-GENAI model/gen-ai/gen-ai-tool-call-arguments.json, written out
# by hand without its title and description annotations.
SCHEMA: dict[str, Any] = {"additionalProperties": True, "type": "object"}


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return node_rows(ATTRIBUTE, Key((address, "")), value)
