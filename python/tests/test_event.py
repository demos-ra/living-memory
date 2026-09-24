"""Tests of _event: how a call is written as the details event."""

import unittest

from living_memory import _event
from living_memory._json import Number


class TestLogsData(unittest.TestCase):
    def test_one_event(self):
        attributes = [{"key": "k", "value": {"stringValue": "v"}}]
        self.assertEqual(
            _event.logs_data(attributes),
            {
                "resourceLogs": [
                    {
                        "scopeLogs": [
                            {
                                "logRecords": [
                                    {
                                        "eventName": (
                                            "gen_ai.client.inference.operation.details"
                                        ),
                                        "attributes": attributes,
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
        )


class TestAttribute(unittest.TestCase):
    def test_value_types(self):
        cases = [
            ("string", "m", {"stringValue": "m"}),
            ("boolean", True, {"boolValue": True}),
            ("int", Number("40.0"), {"intValue": "40"}),
            ("double", Number("1"), {"doubleValue": "1"}),
            ("string[]", ["a"], {"arrayValue": {"values": [{"stringValue": "a"}]}}),
            ("any", None, {}),
        ]
        for value_type, value, expected in cases:
            with self.subTest(value_type):
                self.assertEqual(
                    _event.attribute("k", value, value_type),
                    {"key": "k", "value": expected},
                )

    def test_not_of_its_value_type(self):
        cases = [
            ("string", Number("1")),
            ("boolean", "true"),
            ("int", Number("1.5")),
            ("int", Number(str(2**63))),
            ("double", Number("1e400")),
            ("string[]", ["a", Number("1")]),
        ]
        for value_type, value in cases:
            with self.subTest(value_type=value_type, value=value):
                with self.assertRaises(ValueError):
                    _event.attribute("k", value, value_type)


class TestTypedAndNamespaced(unittest.TestCase):
    def test_absent_without_a_value(self):
        self.assertEqual(_event.typed("k", None, "string"), [])

    def test_namespaced(self):
        self.assertEqual(
            _event.namespaced("anthropic.request.", {"metadata": "x"}),
            [{"key": "anthropic.request.metadata", "value": {"stringValue": "x"}}],
        )


if __name__ == "__main__":
    unittest.main()
