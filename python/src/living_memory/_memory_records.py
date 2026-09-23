"""The memory records schema.

OTEL-GENAI model/gen-ai/gen-ai-memory-records.json: an array of
MemoryRecord.

Functions:
entries -- return the rows of a gen_ai.memory.records value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
RECORD -- MemoryRecord
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "RECORD", "SHEETS"]

from typing import Any

from living_memory import _relations
from living_memory._relations import Shape

ATTRIBUTE = "gen_ai.memory.records"
RECORD = Shape(
    columns=("id", "score"),
    lines=frozenset({"id"}),
    nodes=("content", "metadata"),
    known=frozenset({"content", "id", "metadata", "score"}),
)
SHEETS = _relations.shape_sheets(ATTRIBUTE, RECORD)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.memory.records value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.array_entries(ATTRIBUTE, RECORD, address, "", value)
