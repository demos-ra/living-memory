"""Tests of _json: how JSON texts are read and JSON strings written."""

import unittest

from living_memory import _json as module
from living_memory._json import Number
from living_memory._json_pointer import PlacedError

# Deeper than any recursion Python allows by default.
DEEP = 5000


class TestDecode(unittest.TestCase):
    # value.6: a number is kept as written.
    def test_numbers_as_written(self):
        value = module.decode(b'{"a": 1.50, "b": 1e2, "c": -0, "d": 2E-3}')
        self.assertEqual(value, {"a": "1.50", "b": "1e2", "c": "-0", "d": "2E-3"})
        for number in value.values():
            with self.subTest(number=number):
                self.assertIsInstance(number, Number)

    # value.5: members are read in the order written.
    def test_members_in_order_written(self):
        self.assertEqual(list(module.decode(b'{"b":1,"a":2}')), ["b", "a"])

    def test_values(self):
        cases = [
            (b" [true, false, null] ", [True, False, None]),
            (b'"a\\"\\\\\\/\\b\\f\\n\\r\\t"', 'a"\\/\b\f\n\r\t'),
            (b'"\\u00e9\\uD834\\uDD1E"', "é\U0001d11e"),
            (b'"\\ud800"', "\ud800"),
            ('"é𝄞"'.encode(), "é𝄞"),
            (b"{}", {}),
            (b"[]", []),
            (b'{"a":[{"b":[]},{}],"c":{}}', {"a": [{"b": []}, {}], "c": {}}),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                self.assertEqual(module.decode(data), expected)

    # value.3: any depth of nesting is read.
    def test_any_depth(self):
        value = module.decode(b"[" * DEEP + b"]" * DEEP)
        for _ in range(DEEP - 1):
            value = value[0]
        self.assertEqual(value, [])

    # value.4, value.7, value.10: a value that is not a JSON text in
    # UTF-8, a byte order mark included, is refused, and placed whole.
    def test_not_a_json_text(self):
        cases = [
            b"x",
            b'{"n":',
            b"[NaN]",
            b"01",
            b"-",
            b"1.",
            b"1e",
            b'"a\nb"',
            b'"\\x"',
            b'"\\u12"',
            b"[1,]",
            b"{,}",
            b'{"a"}',
            b"[1 2]",
            b"1 2",
            b"",
            b"\xef\xbb\xbf{}",
            b'"\xff"',
            b'"\xc0\x80"',
            b'"\xed\xa0\x80"',
            b'"\xf4\x90\x80\x80"',
            b'"\xe2\x82"',
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(PlacedError) as raised:
                    module.decode(data)
                self.assertEqual(raised.exception.pointer, "")

    # value.8, value.10: the first object, in the order written, whose
    # names are not all unique is named by its pointer.
    def test_repeated_name(self):
        cases = [
            (b'{"n":1,"n":2}', ""),
            (b'{"a":[{"x":1},{"y":1,"y":2}]}', "/a/1"),
            (b'{"a":{"x":1,"x":2},"a":3}', ""),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                with self.assertRaises(PlacedError) as raised:
                    module.decode(data)
                self.assertEqual(raised.exception.pointer, expected)


class TestEncodeString(unittest.TestCase):
    # field.5: a pointer is reported as a JSON string.
    def test_escapes(self):
        cases = [
            ("/0/t", '"/0/t"'),
            ("/0/a\tb", '"/0/a\\tb"'),
            ('/0/"\\', '"/0/\\"\\\\"'),
            ("/0/\x01", '"/0/\\u0001"'),
            ("/0/\ud800", '"/0/\\ud800"'),
            ("/0/é/a/b", '"/0/é/a/b"'),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(module.encode_string(value), expected)


class TestEncode(unittest.TestCase):
    # value.5, value.6: a value written back is the text read, members
    # in order and numbers as written.
    def test_round_trip(self):
        text = b'{"b":[1.50,true,null],"a":"x\\ty","c":{}}'
        self.assertEqual(module.encode(module.decode(text)), text)

    # With an indent, each member and element on a line of its own.
    def test_indent(self):
        text = b'{\n  "a": [\n    1,\n    {}\n  ],\n  "b": []\n}'
        self.assertEqual(module.encode(module.decode(text), 2), text)

    # value.3: any depth of nesting is written.
    def test_any_depth(self):
        text = b"[" * DEEP + b"]" * DEEP
        self.assertEqual(module.encode(module.decode(text)), text)


class TestType(unittest.TestCase):
    def test_six_primitive_types(self):
        cases = [({}, "object"), ([], "array"), ("x", "string")]
        cases += [(Number("1"), "number"), (True, "boolean"), (None, "null")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(module.primitive_type(value), expected)
        self.assertEqual(set(module.TYPES), {expected for _, expected in cases})


if __name__ == "__main__":
    unittest.main()
