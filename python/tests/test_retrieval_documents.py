"""Tests of _retrieval_documents: the retrieval documents schema."""

import unittest

from living_memory import _retrieval_documents as module
from living_memory._relations import Number

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestRetrievalDocuments(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [{"id": "doc_123", "score": Number("0.950")}, {"id": "d", "title": "T"}]
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.retrieval.documents", [A, "/0", "doc_123", "0.950"]),
                ("gen_ai.retrieval.documents", [A, "/1", "d", ""]),
                (
                    "gen_ai.retrieval.documents.additionalProperties",
                    [A, "/1/title", "string", "T"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
