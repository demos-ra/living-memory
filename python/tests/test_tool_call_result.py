"""Tests of _tool_call_result: the tool call result schema."""

import unittest

from living_memory import _tool_call_result as module
from living_memory._json_schema import validates

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = {"conditions": "first\nsecond"}


class TestToolCallResult(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(validates(VALUE, module.SCHEMA, module.SCHEMA))
        self.assertFalse(validates("x", module.SCHEMA, module.SCHEMA))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.tool.call.result", [A, "", "object", ""]),
                (
                    "gen_ai.tool.call.result",
                    [A, "/conditions", "string", "first\nsecond"],
                ),
                ("gen_ai.tool.call.result.value", [A, "/conditions", "0", "first"]),
                ("gen_ai.tool.call.result.value", [A, "/conditions", "1", "second"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
