"""Tests of _parts: the message part types."""

import unittest

from living_memory import _parts, _relations

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


class TestSheets(unittest.TestCase):
    """The parts' sheets are the spec's, in its order."""

    def test_message_parts(self):
        for prefix in ("gen_ai.input.messages.parts", "gen_ai.output.messages.parts"):
            with self.subTest(prefix):
                self.assertEqual(
                    _relations.variant_sheets(prefix, _parts.MESSAGE_PARTS),
                    spec_sheets(prefix + "."),
                )


class TestEntries(unittest.TestCase):
    """A part's rows follow its definition."""

    def test_server_tool_call(self):
        value = [
            {
                "type": "server_tool_call",
                "id": "s2",
                "name": "code",
                "server_tool_call": {"type": "code_interpreter", "code": "print(1)"},
            }
        ]
        self.assertEqual(
            _relations.variant_entries("p", _parts.MESSAGE_PARTS, A, "/0/parts", value),
            [
                ("p.server_tool_call", [A, "/0/parts/0", "s2", "code"]),
                (
                    "p.server_tool_call.server_tool_call",
                    [A, "/0/parts/0/server_tool_call", "code_interpreter"],
                ),
                (
                    "p.server_tool_call.server_tool_call.additionalProperties",
                    [A, "/0/parts/0/server_tool_call/code", "string", "print(1)"],
                ),
            ],
        )

    def test_blob_has_further_properties(self):
        value = [{"type": "blob", "modality": "image", "content": "x", "name": "logo"}]
        self.assertEqual(
            _relations.variant_entries("p", _parts.MESSAGE_PARTS, A, "/0/parts", value),
            [
                ("p.blob", [A, "/0/parts/0", "", "image", "x"]),
                (
                    "p.blob.additionalProperties",
                    [A, "/0/parts/0/name", "string", "logo"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
