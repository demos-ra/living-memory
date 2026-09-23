"""The retrieval documents schema.

OTEL-GENAI model/gen-ai/gen-ai-retrieval-documents.json: an array of
RetrievalDocument.

Functions:
entries -- return the rows of a gen_ai.retrieval.documents value

Constants:
ATTRIBUTE -- the attribute's key, and the name its sheets start from
DOCUMENT -- RetrievalDocument
SHEETS -- the set's sheets, in order, with their headers
"""

__all__ = ["entries", "ATTRIBUTE", "DOCUMENT", "SHEETS"]

from typing import Any

from living_memory import _relations
from living_memory._relations import Shape

ATTRIBUTE = "gen_ai.retrieval.documents"
DOCUMENT = Shape(
    columns=("id", "score"),
    lines=frozenset({"id"}),
    known=frozenset({"id", "score"}),
)
SHEETS = _relations.shape_sheets(ATTRIBUTE, DOCUMENT)


def entries(address: str, value: Any) -> list[tuple[str, list[str]]]:
    """Return the rows of a gen_ai.retrieval.documents value.

    address -- the address of the span or log record that holds it
    value -- the decoded attribute value
    """
    return _relations.array_entries(ATTRIBUTE, DOCUMENT, address, "", value)
