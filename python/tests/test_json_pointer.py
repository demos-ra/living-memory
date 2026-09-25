"""Tests of _json_pointer: RFC 6901."""

import unittest

from living_memory import _json_pointer as module


class TestPointer(unittest.TestCase):
    # record.3: '~' is written '~0' and '/' is written '~1'.
    def test_tokens_escaped(self):
        self.assertEqual(module.pointer("/0", "a~b/c"), "/0/a~0b~1c")
        self.assertEqual(module.pointer("", 3), "/3")


class TestPlacedError(unittest.TestCase):
    # module.1, value.3: a failure names its place by its pointer.
    def test_fields(self):
        error = module.PlacedError("a name is repeated", "/a")
        self.assertIsInstance(error, ValueError)
        self.assertEqual((error.msg, error.pointer), ("a name is repeated", "/a"))


class TestEvaluate(unittest.TestCase):
    # file.1, module.3: a token names a member or an element.
    def test_members_and_elements(self):
        document = {"a/b": [1, {"~": 2}]}
        self.assertEqual(module.evaluate(document, "/a~1b/1/~0"), 2)
        self.assertEqual(module.evaluate(document, ""), document)


if __name__ == "__main__":
    unittest.main()
