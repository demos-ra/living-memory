"""Tests of _tool_definitions: the tool definitions schema."""

import unittest

from living_memory import _tool_definitions as module
from living_memory._json_schema import validates
from living_memory._json import decode

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = [
    {"type": "function", "name": "f", "parameters": {"required": ["a"]}},
    {"type": "web_search", "name": "search"},
]


class TestToolDefinitions(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(validates(VALUE, module.SCHEMA, module.SCHEMA))
        invalid = [{"type": "web_search"}]
        self.assertFalse(validates(invalid, module.SCHEMA, module.SCHEMA))

    def test_invalid_function_is_generic(self):
        value = decode('[{"type": "function", "name": "f", "parameters": 1}]')
        self.assertTrue(validates(value, module.SCHEMA, module.SCHEMA))
        self.assertEqual(
            module.rows(A, value),
            [
                ("gen_ai.tool.definitions.generic", [A, "/0", "function", "f"]),
                (
                    "gen_ai.tool.definitions.generic.additionalProperties",
                    [A, "/0/parameters", "number", "1"],
                ),
            ],
        )

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.tool.definitions.function", [A, "/0", "f", ""]),
                (
                    "gen_ai.tool.definitions.function.parameters",
                    [A, "/0/parameters", "object", ""],
                ),
                (
                    "gen_ai.tool.definitions.function.parameters",
                    [A, "/0/parameters/required", "array", ""],
                ),
                (
                    "gen_ai.tool.definitions.function.parameters",
                    [A, "/0/parameters/required/0", "string", "a"],
                ),
                ("gen_ai.tool.definitions.generic", [A, "/1", "web_search", "search"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
