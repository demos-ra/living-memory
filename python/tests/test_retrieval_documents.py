"""Tests of _retrieval_documents: the retrieval documents schema."""

import unittest

from living_memory import _retrieval_documents as module
from living_memory._json import Number

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = [{"id": "doc_123", "score": Number("0.950")}, {"id": "d", "title": "T"}]


class TestRetrievalDocuments(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.sheets(), spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(module.validates(VALUE))
        self.assertFalse(module.validates([{"score": "high"}]))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
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
