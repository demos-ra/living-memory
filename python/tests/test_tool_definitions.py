"""Tests of _tool_definitions: the tool definitions schema."""

import unittest

from living_memory import _tool_definitions as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestToolDefinitions(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [
            {"type": "function", "name": "f", "parameters": {"required": ["a"]}},
            {"type": "web_search", "name": "search"},
        ]
        self.assertEqual(
            module.entries(A, value),
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
