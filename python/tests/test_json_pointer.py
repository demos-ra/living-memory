"""Tests of _json_pointer: a JSON Pointer, written and evaluated."""

import unittest

from living_memory import _json_pointer


class TestPointer(unittest.TestCase):
    def test_escapes_tilde_and_slash(self):
        self.assertEqual(_json_pointer.pointer("", "a/b~c"), "/a~1b~0c")
        self.assertEqual(_json_pointer.pointer("/0", 2), "/0/2")


class TestEvaluate(unittest.TestCase):
    # RFC 6901, Section 5: the example document and its pointers.
    DOCUMENT = {
        "foo": ["bar", "baz"],
        "": 0,
        "a/b": 1,
        "c%d": 2,
        "m~n": 8,
    }

    def test_examples(self):
        cases = [
            ("", self.DOCUMENT),
            ("/foo", ["bar", "baz"]),
            ("/foo/0", "bar"),
            ("/", 0),
            ("/a~1b", 1),
            ("/c%d", 2),
            ("/m~0n", 8),
        ]
        for at, expected in cases:
            with self.subTest(at):
                self.assertEqual(_json_pointer.evaluate(self.DOCUMENT, at), expected)

    def test_tilde_one_first(self):
        # '~01' is '~1', not '/' (RFC 6901, Section 4).
        self.assertEqual(_json_pointer.evaluate({"~1": "x"}, "/~01"), "x")


if __name__ == "__main__":
    unittest.main()
