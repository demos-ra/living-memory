"""Tests of _protojson: how an OTLP simple value is written in JSON."""

import unittest

from living_memory._json import Number
from living_memory._protojson import valid


class TestValid(unittest.TestCase):
    def test_types(self):
        cases = [
            ("x", "string", "name", True),
            (Number("1"), "string", "name", False),
            (True, "bool", "isMonotonic", True),
            ("true", "bool", "isMonotonic", False),
            ("5B8EFFF798038103D269B633813FC60C", "bytes", "traceId", True),
            ("zz", "bytes", "spanId", False),
            ("abc", "bytes", "spanId", False),
            ("AAECAwQFBgc=", "bytes", "parentSpanId", True),
            ("AAECAwQFBgc", "bytes", "bytesValue", True),
            ("-_8", "bytes", "bytesValue", True),
            ("!!", "bytes", "bytesValue", False),
            ("A", "bytes", "bytesValue", False),
            (Number("0.5"), "double", "asDouble", True),
            ("NaN", "double", "asDouble", True),
            ("1e3", "double", "asDouble", True),
            ("", "double", "asDouble", False),
            (Number("2"), "enum", "kind", True),
            ("2", "enum", "kind", False),
            ("SPAN_KIND_CLIENT", "enum", "kind", False),
            (Number("1.0"), "uint32", "flags", True),
            (Number("1.5"), "uint32", "flags", False),
            (Number("-1"), "uint32", "flags", False),
            ("4294967296", "uint32", "flags", False),
            ("1581452772000000321", "fixed64", "timeUnixNano", True),
            ("1e2", "fixed64", "timeUnixNano", True),
            ("", "fixed64", "timeUnixNano", False),
            ("-5", "sfixed64", "asInt", True),
            ("-5", "fixed64", "timeUnixNano", False),
        ]
        for value, proto, name, expected in cases:
            with self.subTest(value=value, proto=proto):
                self.assertIs(valid(value, proto, name), expected)


if __name__ == "__main__":
    unittest.main()
