"""Tests of _input_messages: the input messages schema."""

import unittest

from living_memory import _input_messages as module

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestInputMessages(unittest.TestCase):
    """The set's sheets and rows follow its schema."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets(module.ATTRIBUTE))

    def test_entries(self):
        value = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": "a\nb"}],
                "name": None,
            },
            {"role": "assistant", "parts": []},
        ]
        self.assertEqual(
            module.entries(A, value),
            [
                ("gen_ai.input.messages", [A, "/0", "user", ""]),
                ("gen_ai.input.messages.parts.text", [A, "/0/parts/0", ""]),
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
