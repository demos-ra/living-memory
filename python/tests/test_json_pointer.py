"""Tests of _json_pointer: RFC 6901."""

import unittest

from living_memory import _json_pointer as module


class TestPointer(unittest.TestCase):
    # key.4: '~' is written '~0' and '/' is written '~1'.
    def test_tokens_escaped(self):
        self.assertEqual(module.pointer("/0", "a~b/c"), "/0/a~0b~1c")
        self.assertEqual(module.pointer("", 3), "/3")


class TestPlacedError(unittest.TestCase):
    # schema.8, value.10: a failure names its place by its pointer.
    def test_fields(self):
        error = module.PlacedError("a name is repeated", "/a")
        self.assertIsInstance(error, ValueError)
        self.assertEqual((error.msg, error.pointer), ("a name is repeated", "/a"))


class TestTokens(unittest.TestCase):
    # sheet.2: a kind's name may be its $ref's last token, unescaped,
    # '~01' being '~1'.
    def test_unescaped(self):
        self.assertEqual(module.tokens("/a~1b/~01"), ["a/b", "~1"])
        self.assertEqual(module.tokens(""), [])


class TestEvaluate(unittest.TestCase):
    # schema.10: a token names a member or an element.
    def test_members_and_elements(self):
        document = {"a/b": [1, {"~": 2}]}
        self.assertEqual(module.evaluate(document, "/a~1b/1/~0"), 2)
        self.assertEqual(module.evaluate(document, ""), document)

    # schema.5: an array index is "0", or digits without a leading "0";
    # any other token fails.
    def test_not_an_index(self):
        for at in ("/01", "/-", "/x", "/", "/1.0"):
            with self.subTest(at=at):
                with self.assertRaises(ValueError):
                    module.evaluate([1, 2], at)
        self.assertEqual(module.evaluate(list(range(11)), "/10"), 10)


class TestFromFragment(unittest.TestCase):
    # schema.6, schema.10: a fragment is percent-decoded as UTF-8, and
    # is a JSON Pointer; any other is refused.
    def test_decoded(self):
        self.assertEqual(module.from_fragment("/a%20b%C3%A9/~1"), "/a bé/~1")
        self.assertEqual(module.from_fragment(""), "")

    def test_refused(self):
        for fragment in ("A", "/a b", "/%2", "/%FF", "/a#"):
            with self.subTest(fragment=fragment):
                with self.assertRaises(ValueError):
                    module.from_fragment(fragment)


if __name__ == "__main__":
    unittest.main()
