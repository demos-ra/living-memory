"""Tests of _separators: what separates MTSV text."""

import unittest

from living_memory import _separators as module


class TestCannotHold(unittest.TestCase):
    # module.1, field.3: a field holds no HT, LF, FF or CR.
    def test_each_character(self):
        for text, expected in [("ab", False), ("a\tb", True), ("a\rb", True)]:
            with self.subTest(text=text):
                self.assertIs(module.cannot_hold(text), expected)


class TestSeparators(unittest.TestCase):
    # field.2: FF, a line break and HT separate sheets, records and
    # fields; a lone CR is none of them.
    def test_holds_separator(self):
        cases = [("a", False), ("a\rb", False), ("a\fb", True), ("a\nb", True)]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(module.holds_separator(text), expected)

    # relation.3: a line break is LF or CRLF.
    def test_split(self):
        self.assertEqual(module.pages("a\fb"), ["a", "b"])
        self.assertEqual(module.lines("a\nb\r\nc"), ["a", "b", "c"])
        self.assertEqual(module.runs("a\tb"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
