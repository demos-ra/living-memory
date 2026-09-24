"""Tests of _memory_records: the memory records schema."""

import unittest

from living_memory import _memory_records as module
from living_memory._json import Number

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = [{"content": "dark mode", "id": "mem_123", "score": Number("0.95")}]


class TestMemoryRecords(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.sheets(), spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(module.validates(VALUE))
        self.assertFalse(module.validates([{"id": "m"}]))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.memory.records", [A, "/0", "mem_123", "0.95"]),
                (
                    "gen_ai.memory.records.content",
                    [A, "/0/content", "string", "dark mode"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
