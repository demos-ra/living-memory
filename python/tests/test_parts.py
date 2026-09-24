"""Tests of _parts: the message part types."""

import unittest

from living_memory import _parts
from living_memory._relations import Key

from support import spec_sheets

A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


def parts(value):
    return _parts.MESSAGE_PARTS.array_rows("p", Key((A, "/0/parts")), value)


class TestSheets(unittest.TestCase):
    def test_message_parts(self):
        for prefix in ("gen_ai.input.messages.parts", "gen_ai.output.messages.parts"):
            with self.subTest(prefix):
                self.assertEqual(
                    _parts.MESSAGE_PARTS.sheets(prefix),
                    spec_sheets(prefix + "."),
                )


class TestDefinitions(unittest.TestCase):
    def test_definitions(self):
        self.assertEqual(
            _parts.DEFINITIONS,
            {
                "blob": "BlobPart",
                "compaction": "CompactionPart",
                "file": "FilePart",
                "reasoning": "ReasoningPart",
                "server_tool_call": "ServerToolCallPart",
                "server_tool_call_response": "ServerToolCallResponsePart",
                "text": "TextPart",
                "tool_call": "ToolCallRequestPart",
                "tool_call_response": "ToolCallResponsePart",
                "uri": "UriPart",
            },
        )
        self.assertEqual(
            set(_parts.DEFINITIONS) | {"generic"}, set(_parts.MESSAGE_PARTS.definitions)
        )


class TestEntries(unittest.TestCase):
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
            parts(value),
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
            parts(value),
            [
                ("p.blob", [A, "/0/parts/0", "", "image", "x"]),
                (
                    "p.blob.additionalProperties",
                    [A, "/0/parts/0/name", "string", "logo"],
                ),
            ],
        )

    def test_invalid_as_named_is_generic(self):
        value = [{"type": "blob", "content": "x"}]
        self.assertEqual(
            parts(value),
            [
                ("p.generic", [A, "/0/parts/0", "blob"]),
                (
                    "p.generic.additionalProperties",
                    [A, "/0/parts/0/content", "string", "x"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
