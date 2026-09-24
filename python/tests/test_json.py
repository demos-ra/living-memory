"""Tests of _json: how a JSON text is read and written."""

import unittest

from living_memory import _json as module
from living_memory._json import Number


class TestDecode(unittest.TestCase):
    def test_numbers_as_written(self):
        value = module.decode('{"a": 0.950, "b": 5, "c": -1e3}')
        self.assertEqual(value, {"a": "0.950", "b": "5", "c": "-1e3"})
        for number in value.values():
            with self.subTest(number):
                self.assertIsInstance(number, Number)

    def test_not_json(self):
        with self.assertRaises(ValueError):
            module.decode("x")

    def test_non_finite_tokens_are_not_json(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token):
                with self.assertRaises(ValueError):
                    module.decode(f"[{token}]")

    def test_last_of_a_repeated_name_in_its_place(self):
        value = module.decode('{"a": 1, "b": 2, "a": 3}')
        self.assertEqual(list(value.items()), [("b", "2"), ("a", "3")])


class TestKind(unittest.TestCase):
    def test_types(self):
        cases = [
            ({}, "object"),
            ([], "array"),
            ("x", "string"),
            (Number("1"), "number"),
            (1, "number"),
            (True, "boolean"),
            (False, "boolean"),
            (None, "null"),
        ]
        for value, expected in cases:
            with self.subTest(value):
                self.assertEqual(module.type(value), expected)


class TestEncode(unittest.TestCase):
    def test_written_as_read(self):
        for text in ('{"a":[1.50,true,null,"é"],"b":{}}', "4e1", '"\\u0009"'):
            with self.subTest(text):
                value = module.decode(text)
                self.assertEqual(module.decode(module.encode(value)), value)
        self.assertEqual(module.encode(module.decode("1.50")), "1.50")


if __name__ == "__main__":
    unittest.main()
