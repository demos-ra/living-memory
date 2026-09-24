"""Tests of _system_instructions: the system instructions schema."""

import unittest

from living_memory import _system_instructions as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = [{"type": "text", "content": "Hi", "lang": "en"}, {"type": "note"}]


class TestSystemInstructions(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.sheets(), spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(module.validates(VALUE))
        self.assertFalse(module.validates([{"content": "Hi"}]))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
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
