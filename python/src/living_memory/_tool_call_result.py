"""The tool call result schema.

OTEL-GENAI model/gen-ai/gen-ai-tool-call-result.json.
"""

__all__ = ["rows", "sheets", "validates", "ATTRIBUTE"]

from typing import Any

from living_memory import _json_schema
from living_memory._relations import Key, node_rows, node_sheets

ATTRIBUTE = "gen_ai.tool.call.result"

# The schema is written out by hand, without its title and
# description annotations (OTEL-GENAI,
# model/gen-ai/gen-ai-tool-call-result.json).
_SCHEMA: dict[str, Any] = {"additionalProperties": True, "type": "object"}


def sheets() -> list[tuple[str, list[str]]]:
    return node_sheets(ATTRIBUTE)


def validates(value: Any) -> bool:
    return _json_schema.validates(value, _SCHEMA, _SCHEMA)


def rows(address: str, value: Any) -> list[tuple[str, list[str]]]:
    return node_rows(ATTRIBUTE, Key((address, "")), value)
