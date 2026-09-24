"""Tests of _input_messages: the input messages schema."""

import unittest

from living_memory import _input_messages as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"
VALUE = [
    {"role": "user", "parts": [{"type": "text", "content": "a\nb"}], "name": None},
    {"role": "assistant", "parts": []},
]


class TestInputMessages(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.sheets(), spec_sheets(module.ATTRIBUTE))

    def test_schema(self):
        self.assertTrue(module.validates(VALUE))
        for invalid in ([{"role": "user"}], [{"role": 1, "parts": []}], {}):
            with self.subTest(invalid):
                self.assertFalse(module.validates(invalid))

    def test_rows(self):
        self.assertEqual(
            module.rows(A, VALUE),
            [
                ("gen_ai.input.messages", [A, "/0", "user", ""]),
                ("gen_ai.input.messages.parts.text", [A, "/0/parts/0", "a\nb"]),
                (
                    "gen_ai.input.messages.parts.text.content",
                    [A, "/0/parts/0/content", "0", "a"],
                ),
                (
                    "gen_ai.input.messages.parts.text.content",
                    [A, "/0/parts/0/content", "1", "b"],
                ),
                ("gen_ai.input.messages", [A, "/1", "assistant", ""]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
