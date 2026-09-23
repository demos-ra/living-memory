"""Tests of _output_messages: the output messages schema."""

import unittest

from living_memory import _output_messages as module

from support import spec_sheets

A = "/0/resourceLogs/0/scopeLogs/0/logRecords/0"


class TestOutputMessages(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Rainy."}],
                "finish_reason": "stop",
            }
        ]
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.output.messages", [A, "/0", "assistant", "", "stop"]),
                ("gen_ai.output.messages.parts.text", [A, "/0/parts/0", "Rainy."]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
