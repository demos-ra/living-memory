"""Tests of _system_instructions: the system instructions schema."""

import unittest

from living_memory import _system_instructions as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestSystemInstructions(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [{"type": "text", "content": "Hi", "lang": "en"}, {"type": "note"}]
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.system_instructions.text", [A, "/0", "Hi"]),
                (
                    "gen_ai.system_instructions.text.additionalProperties",
                    [A, "/0/lang", "string", "en"],
                ),
                ("gen_ai.system_instructions.generic", [A, "/1", "note"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
