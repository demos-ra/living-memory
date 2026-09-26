"""Tests of _field: a value as the text of a field."""

import unittest

from living_memory import _field as module
from living_memory._json import Number


class TestText(unittest.TestCase):
    # field.1: a field is text.
    def test_values(self):
        cases = [("x", "x"), (Number("1.50"), "1.50"), (True, "true")]
        cases += [(False, "false"), (None, ""), ({"a": "x"}, ""), (["x"], "")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(module.text(value), expected)

    # field.2: a string holding FF, a line break or HT is an empty
    # field.
    def test_separators_empty(self):
        for value in ("a\tb", "a\nb", "a\fb", "a\r\nb"):
            with self.subTest(value=value):
                self.assertEqual(module.text(value), "")

    # field.1: the type of an instance is its primitive type.
    def test_type(self):
        self.assertEqual(module.type(Number("1")), "number")
        self.assertEqual(module.type({}), "object")


class TestCarried(unittest.TestCase):
    # field.3: a lone CR and a character that is not text are left out.
    def test_lone_cr_and_surrogate(self):
        self.assertEqual(module.carried("a\rb"), ("ab", True))
        self.assertEqual(module.carried("a\ud800b"), ("ab", True))
        self.assertEqual(module.carried("a\r\nb"), ("a\r\nb", False))

    # field.3: a member name within a pointer loses HT, LF, FF and CR.
    def test_names(self):
        for name in ("a\tb", "a\nb", "a\fb", "a\rb", "a\ud800b"):
            with self.subTest(name=name):
                self.assertEqual(module.name_carried(name), ("ab", True))
        self.assertEqual(module.name_carried("ab"), ("ab", False))


if __name__ == "__main__":
    unittest.main()
