"""Tests of _json_pointer: RFC 6901."""

import unittest

from living_memory import _json_pointer as module


class TestPointer(unittest.TestCase):
    # key.1: '~' is written '~0' and '/' is written '~1'.
    def test_tokens_escaped(self):
        self.assertEqual(module.pointer("/0", "a~b/c"), "/0/a~0b~1c")
        self.assertEqual(module.pointer("", 3), "/3")


class TestPlacedError(unittest.TestCase):
    # module.1, value.3: a failure names its place by its pointer.
    def test_fields(self):
        error = module.PlacedError("a name is repeated", "/a")
        self.assertIsInstance(error, ValueError)
        self.assertEqual((error.msg, error.pointer), ("a name is repeated", "/a"))


class TestTokens(unittest.TestCase):
    # sheet.1: a kind's name may be its $ref's last token, unescaped,
    # '~01' being '~1'.
    def test_unescaped(self):
        self.assertEqual(module.tokens("/a~1b/~01"), ["a/b", "~1"])
        self.assertEqual(module.tokens(""), [])


class TestEvaluate(unittest.TestCase):
    # module.2: a token names a member or an element.
    def test_members_and_elements(self):
        document = {"a/b": [1, {"~": 2}]}
        self.assertEqual(module.evaluate(document, "/a~1b/1/~0"), 2)
        self.assertEqual(module.evaluate(document, ""), document)


if __name__ == "__main__":
    unittest.main()
