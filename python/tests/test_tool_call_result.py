"""Tests of _tool_call_result: the tool call result schema."""

import unittest

from living_memory import _tool_call_result as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestToolCallResult(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = {"conditions": "first\nsecond"}
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.tool.call.result", [A, "", "object", ""]),
                ("gen_ai.tool.call.result", [A, "/conditions", "string", ""]),
                ("gen_ai.tool.call.result.value", [A, "/conditions", "0", "first"]),
                ("gen_ai.tool.call.result.value", [A, "/conditions", "1", "second"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
