"""Tests of _sets: which attributes are the eight, and their reading."""

import unittest

from living_memory import _sets

from support import spec_sheets

EIGHT = [
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.memory.records",
    "gen_ai.retrieval.documents",
]
TEXT = '[{"type":"text","content":"Be brief."}]'


class TestIsSet(unittest.TestCase):
    def test_the_eight(self):
        for key in EIGHT:
            with self.subTest(key):
                self.assertTrue(_sets.is_set(key))
        self.assertFalse(_sets.is_set("gen_ai.request.model"))


class TestSheets(unittest.TestCase):
    def test_in_the_order_of_the_sheets(self):
        found = [name for name, _ in _sets.sheets()]
        expected = [name for name, _ in spec_sheets("gen_ai.")]
        self.assertEqual(found, expected)


class TestRows(unittest.TestCase):
    def test_a_json_string_and_its_structured_form(self):
        structured = {
            "arrayValue": {
                "values": [
                    {
                        "kvlistValue": {
                            "values": [
                                {"key": "type", "value": {"stringValue": "text"}},
                                {
                                    "key": "content",
                                    "value": {"stringValue": "Be brief."},
                                },
                            ]
                        }
                    }
                ]
            }
        }
        key = "gen_ai.system_instructions"
        expected = [("gen_ai.system_instructions.text", ["/a", "/0", "Be brief."])]
        self.assertEqual(_sets.rows(key, "/a", {"stringValue": TEXT}), expected)
        self.assertEqual(_sets.rows(key, "/a", structured), expected)


class TestCheck(unittest.TestCase):
    def test_conforming_and_not(self):
        key = "gen_ai.system_instructions"
        _sets.check(key, {"stringValue": TEXT}, "/a")
        for value in ({"stringValue": "x"}, {"stringValue": '[{"content":"Hi"}]'}):
            with self.subTest(value):
                with self.assertRaises(ValueError):
                    _sets.check(key, value, "/a")


if __name__ == "__main__":
    unittest.main()
