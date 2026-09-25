"""Tests of _fields: text in MTSV fields."""

import unittest

from living_memory import _fields as module
from living_memory._json import Number


class TestText(unittest.TestCase):
    def test_values(self):
        cases = [("x", "x"), (Number("1.50"), "1.50"), (True, "true")]
        cases += [(False, "false"), (None, ""), ({"a": "x"}, ""), (["x"], "")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(module.text(value), expected)


class TestCarried(unittest.TestCase):
    def test_lone_cr_and_surrogate_left_out(self):
        self.assertEqual(module.carried("a\rb"), ("ab", True))
        self.assertEqual(module.carried("a\ud800b"), ("ab", True))

    def test_crlf_kept(self):
        self.assertEqual(module.carried("a\r\nb"), ("a\r\nb", False))

    def test_names_lose_separators(self):
        self.assertEqual(module.name_carried("a\tb"), ("ab", True))
        self.assertEqual(module.name_carried("ab"), ("ab", False))


class TestRuns(unittest.TestCase):
    def test_holds_separator(self):
        for value, expected in [("a", False), ("a\tb", True), ("\n", True)]:
            with self.subTest(value=value):
                self.assertEqual(module.holds_separator(value), expected)

    def test_pages_lines_positions(self):
        found = module.runs("a\tb\nc\r\nd\fe")
        expected = [(0, 0, 0, "a"), (0, 0, 1, "b"), (0, 1, 0, "c")]
        expected += [(0, 2, 0, "d"), (1, 0, 0, "e")]
        self.assertEqual(found, expected)

    def test_empty_runs(self):
        self.assertEqual(module.runs("\n"), [(0, 0, 0, ""), (0, 1, 0, "")])


if __name__ == "__main__":
    unittest.main()
