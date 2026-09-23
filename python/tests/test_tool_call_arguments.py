"""Tests of _tool_call_arguments: the tool call arguments schema."""

import unittest

from living_memory import _tool_call_arguments as module
from living_memory._relations import Number

from support import spec_sheets

A = "/2/resourceSpans/0/scopeSpans/0/spans/0"


class TestToolCallArguments(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = {"location": "Paris", "days": [Number("1")]}
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.tool.call.arguments", [A, "", "object", ""]),
                ("gen_ai.tool.call.arguments", [A, "/location", "string", "Paris"]),
                ("gen_ai.tool.call.arguments", [A, "/days", "array", ""]),
                ("gen_ai.tool.call.arguments", [A, "/days/0", "number", "1"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
