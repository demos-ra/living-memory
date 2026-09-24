"""Tests of _any_value: an AnyValue and the JSON value it maps to."""

import unittest

from living_memory import _any_value
from living_memory._json import Number, decode


class TestRepresent(unittest.TestCase):
    def test_each_member(self):
        cases = [
            ({"stringValue": "x"}, "x"),
            ({"boolValue": False}, False),
            ({"intValue": "5"}, Number("5")),
            ({"doubleValue": Number("0.5")}, Number("0.5")),
            ({"doubleValue": "NaN"}, "NaN"),
            ({"bytesValue": "aGk="}, "aGk="),
            ({}, None),
            ({"stringValueStrindex": Number("3")}, None),
            ({"arrayValue": {"values": [{"stringValue": "p"}]}}, ["p"]),
            (
                {"kvlistValue": {"values": [{"key": "k", "value": {"intValue": "1"}}]}},
                {"k": Number("1")},
            ),
        ]
        for value, expected in cases:
            with self.subTest(value):
                self.assertEqual(_any_value.represent(value), expected)

    def test_the_last_member_of_the_oneof(self):
        value = {"intValue": "1", "stringValue": "last"}
        self.assertEqual(_any_value.represent(value), "last")


class TestConvert(unittest.TestCase):
    def test_each_type(self):
        value = decode('{"i": 7, "big": 18446744073709551616, "d": 1.5}')
        cases = [
            (value["i"], {"intValue": "7"}),
            (value["big"], {"stringValue": "18446744073709551616"}),
            (value["d"], {"doubleValue": "1.5"}),
            (True, {"boolValue": True}),
            (None, {}),
            ([], {"arrayValue": {"values": []}}),
            ("s", {"stringValue": "s"}),
            ({"a": None}, {"kvlistValue": {"values": [{"key": "a", "value": {}}]}}),
        ]
        for json_value, expected in cases:
            with self.subTest(json_value):
                self.assertEqual(_any_value.convert(json_value), expected)

    def test_an_int_as_the_decimal_string_of_its_value(self):
        for written in ("40.0", "4e1", "40"):
            with self.subTest(written):
                self.assertEqual(
                    _any_value.convert(Number(written)), {"intValue": "40"}
                )


class TestRanges(unittest.TestCase):
    def test_int64_and_double(self):
        self.assertTrue(_any_value.is_int64(Number(str(2**63 - 1))))
        self.assertFalse(_any_value.is_int64(Number(str(2**63))))
        self.assertFalse(_any_value.is_int64(Number("1.5")))
        self.assertFalse(_any_value.is_int64(True))
        self.assertTrue(_any_value.is_double(Number("1e308")))
        self.assertFalse(_any_value.is_double(Number("1e400")))


class TestMembers(unittest.TestCase):
    def test_members_and_values(self):
        self.assertEqual(
            [name for name, _ in _any_value.members()],
            [
                "stringValue",
                "boolValue",
                "intValue",
                "doubleValue",
                "arrayValue",
                "kvlistValue",
                "bytesValue",
                "stringValueStrindex",
            ],
        )
        self.assertEqual(_any_value.values({"values": [1]}), [1])
        self.assertEqual(_any_value.values({"values": None}), [])
        self.assertEqual(_any_value.winner({"stringValue": None}), ("", None))


if __name__ == "__main__":
    unittest.main()
