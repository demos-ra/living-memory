"""Tests of _tool_call_arguments: the tool call arguments schema."""

import unittest

from living_memory import _tool_call_arguments as module
from living_memory._json import Number

from support import spec_sheets

A = "/2/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = {"location": "Paris", "days": [Number("1")]}


class TestToolCallArguments(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.sheets(), spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(module.validates(VALUE))
        self.assertFalse(module.validates([]))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.tool.call.arguments", [A, "", "object", ""]),
                ("gen_ai.tool.call.arguments", [A, "/location", "string", "Paris"]),
                ("gen_ai.tool.call.arguments", [A, "/days", "array", ""]),
                ("gen_ai.tool.call.arguments", [A, "/days/0", "number", "1"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
