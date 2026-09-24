"""Tests of _fields: how a value is written as the text of a field."""

import unittest

from living_memory import _fields
from living_memory._json import Number


class TestText(unittest.TestCase):
    def test_text(self):
        cases = [
            ("x", "x"),
            (Number("0.950"), "0.950"),
            (True, "true"),
            (False, "false"),
            (None, ""),
            ({}, ""),
            ([], ""),
        ]
        for value, expected in cases:
            with self.subTest(repr(value)):
                self.assertEqual(_fields.text(value), expected)


class TestLines(unittest.TestCase):
    def test_split_at_lf_and_crlf_only(self):
        cases = [
            ("first\nsecond", ["first", "second"]),
            ("first\r\nsecond", ["first", "second"]),
            ("a\n", ["a", ""]),
            ("a\rb", []),
            ("x", []),
            (None, []),
        ]
        for value, expected in cases:
            with self.subTest(repr(value)):
                self.assertEqual(_fields.lines(value), expected)


class TestField(unittest.TestCase):
    def test_what_a_field_cannot_hold_is_left_empty(self):
        for value in ("a\tb", "a\nb", "a\fb", "a\rb"):
            with self.subTest(repr(value)):
                self.assertEqual(_fields.field(value), "")
        self.assertEqual(_fields.field("a b"), "a b")


if __name__ == "__main__":
    unittest.main()
