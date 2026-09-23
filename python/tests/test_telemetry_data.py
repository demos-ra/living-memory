"""Tests of _telemetry_data: the OTLP data tree."""

import unittest

from living_memory import _telemetry_data as module
from living_memory._relations import Number, decode

from support import spec_sheets

SPAN = {
    "traceId": "5b8efff798038103d269b633813fc60c",
    "spanId": "eee19b7ec3c1b174",
    "name": "chat",
    "kind": Number("3"),
    "startTimeUnixNano": "1581452772000000321",
    "endTimeUnixNano": "1581452773000000789",
}


def traces(span, scope=None):
    """Return a TracesData holding one span, and a scope if given."""
    scope_spans = {"scope": scope or {}, "spans": [span]}
    return {"resourceSpans": [{"resource": {}, "scopeSpans": [scope_spans]}]}


class TestSheets(unittest.TestCase):
    """Every sheet of the output is the spec's, in its order."""

    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets())


class TestAnyValue(unittest.TestCase):
    """any_value maps an AnyValue to its JSON value."""

    def test_values(self):
        cases = [
            ({"stringValue": "x"}, "x"),
            ({"boolValue": False}, False),
            ({"intValue": "5"}, "5"),
            ({"doubleValue": Number("0.5")}, "0.5"),
            ({"bytesValue": "aGk="}, "aGk="),
            ({}, None),
            ({"stringValueStrindex": Number("3")}, None),
            ({"arrayValue": {"values": [{"stringValue": "p"}]}}, ["p"]),
            (
                {"kvlistValue": {"values": [{"key": "k", "value": {"intValue": "1"}}]}},
                {"k": "1"},
            ),
            ("x", None),
        ]
        for value, expected in cases:
            with self.subTest(value):
                self.assertEqual(module.any_value(value), expected)

    def test_numbers_are_numbers(self):
        for value in ({"intValue": "5"}, {"doubleValue": Number("0.5")}):
            with self.subTest(value):
                self.assertIsInstance(module.any_value(value), Number)


class TestEntries(unittest.TestCase):
    """entries writes the tree, and the eight in their own sheets."""

    def test_the_eight_on_a_span(self):
        span = dict(SPAN)
        span["attributes"] = [
            {"key": "gen_ai.request.model", "value": {"stringValue": "m"}},
            {
                "key": "gen_ai.system_instructions",
                "value": {"stringValue": '[{"type":"text","content":"Be brief."}]'},
            },
        ]
        a = "/1/resourceSpans/0/scopeSpans/0/spans/0"
        found = module.entries(1, traces(span))
        self.assertIn(
            (
                "resourceSpans.scopeSpans.spans.attributes",
                [a, "gen_ai.request.model", "", "string", "m"],
            ),
            found,
        )
        self.assertIn(
            ("gen_ai.system_instructions.text", [a, "/0", "Be brief."]), found
        )
        keys = [row[1] for name, row in found if name.endswith("spans.attributes")]
        self.assertNotIn("gen_ai.system_instructions", keys)

    def test_the_eight_on_a_scope_are_ordinary(self):
        pair = {"key": "gen_ai.system_instructions", "value": {"stringValue": "x"}}
        scope = {"attributes": [pair]}
        found = module.entries(0, traces(SPAN, scope))
        c = "/0/resourceSpans/0/scopeSpans/0/scope"
        self.assertIn(
            (
                "resourceSpans.scopeSpans.scope.attributes",
                [c, "gen_ai.system_instructions", "", "string", "x"],
            ),
            found,
        )
        self.assertFalse([e for e in found if e[0].startswith("gen_ai.")])

    def test_structure_rows(self):
        found = module.entries(0, traces(SPAN))
        self.assertEqual(
            [name for name, _ in found],
            [
                "resourceSpans",
                "resourceSpans.resource",
                "resourceSpans.scopeSpans",
                "resourceSpans.scopeSpans.scope",
                "resourceSpans.scopeSpans.spans",
            ],
        )
        self.assertEqual(
            found[-1][1],
            [
                "/0/resourceSpans/0/scopeSpans/0/spans/0",
                "5b8efff798038103d269b633813fc60c",
                "eee19b7ec3c1b174",
                "",
                "",
                "",
                "chat",
                "3",
                "1581452772000000321",
                "1581452773000000789",
                "",
                "",
                "",
            ],
        )

    def test_absent_resource_has_no_row(self):
        data = traces(SPAN)
        del data["resourceSpans"][0]["resource"]
        names = [name for name, _ in module.entries(0, data)]
        self.assertNotIn("resourceSpans.resource", names)

    def test_body(self):
        pair = {"key": "msg", "value": {"stringValue": "hi"}}
        record = {"body": {"kvlistValue": {"values": [pair]}}}
        data = {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}
        a = "/0/resourceLogs/0/scopeLogs/0/logRecords/0"
        self.assertEqual(
            [e for e in module.entries(0, data) if e[0].endswith(".body")],
            [
                ("resourceLogs.scopeLogs.logRecords.body", [a, "", "object", ""]),
                ("resourceLogs.scopeLogs.logRecords.body", [a, "/msg", "string", "hi"]),
            ],
        )

    def test_metrics_yield_no_rows(self):
        self.assertEqual(module.entries(0, decode('{"resourceMetrics":[{}]}')), [])

    def test_undecodable_json_string_yields_no_rows(self):
        found = module.set_entries("gen_ai.input.messages", "/0", {"stringValue": "x"})
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
