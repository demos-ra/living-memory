"""Tests of _output_messages: the output messages schema."""

import unittest

from living_memory import _output_messages as module
from living_memory._json_schema import validates

from support import spec_sheets

A = "/0/resourceLogs/0/scopeLogs/0/logRecords/0"
VALUE = [
    {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Rainy."}],
        "finish_reason": "stop",
    }
]


class TestOutputMessages(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(validates(VALUE, module.SCHEMA, module.SCHEMA))
        invalid = [{"role": "assistant", "parts": [], "finish_reason": 1}]
        self.assertFalse(validates(invalid, module.SCHEMA, module.SCHEMA))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.output.messages", [A, "/0", "assistant", "", "stop"]),
                ("gen_ai.output.messages.parts.text", [A, "/0/parts/0", "Rainy."]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
