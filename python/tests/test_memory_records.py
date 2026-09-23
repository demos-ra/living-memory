"""Tests of _memory_records: the memory records schema."""

import unittest

from living_memory import _memory_records as module
from living_memory._relations import Number

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestMemoryRecords(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [{"content": "dark mode", "id": "mem_123", "score": Number("0.95")}]
        self.assertEqual(
            module.entries(A, value),
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
