"""Tests of _telemetry_data: the OTLP data tree."""

import unittest

from living_memory import _telemetry_data as module
from living_memory._json import Number, decode

from support import spec_sheets

SPAN = {
    "traceId": "5b8efff798038103d269b633813fc60c",
    "spanId": "eee19b7ec3c1b174",
    "name": "chat",
    "kind": Number("3"),
    "startTimeUnixNano": "1581452772000000321",
    "endTimeUnixNano": "1581452773000000789",
}
A = "/0/resourceSpans/0/scopeSpans/0/spans/0"


def traces(span, scope=None):
    scope_spans = {"scope": scope or {}, "spans": [span]}
    return {"resourceSpans": [{"resource": {}, "scopeSpans": [scope_spans]}]}


def with_attributes(*pairs):
    return dict(SPAN, attributes=list(pairs))


def attribute_rows(value):
    found = module.rows(0, traces(with_attributes({"key": "k", "value": value})))
    sheet = "resourceSpans.scopeSpans.spans.attributes"
    return [row[2:] for name, row in found if name == sheet]


class TestSheets(unittest.TestCase):
    def test_sheets(self):
        self.assertEqual(module.SHEETS, spec_sheets())


class TestCheck(unittest.TestCase):
    def test_conforming(self):
        cases = [
            traces(SPAN),
            decode('{"resourceMetrics":[{}]}'),
            {"unknown": 1},
            traces(dict(SPAN, droppedAttributesCount="1e1", flags=None)),
            traces(dict(SPAN, parentSpanId="AAECAw==")),
        ]
        for data in cases:
            with self.subTest(data):
                module.check(data)

    def test_non_conforming(self):
        cases = [
            [],
            {"resourceSpans": {}},
            {"resourceSpans": [None]},
            traces(dict(SPAN, traceId="xyz")),
            traces(dict(SPAN, kind="SPAN_KIND_CLIENT")),
            traces(dict(SPAN, kind=Number("1.5"))),
            traces(dict(SPAN, droppedAttributesCount=Number("-1"))),
            traces(dict(SPAN, name=Number("1"))),
            traces(with_attributes({"key": "k", "value": {"doubleValue": "big"}})),
            traces(
                with_attributes(
                    {"key": "gen_ai.input.messages", "value": {"stringValue": "x"}}
                )
            ),
            traces(
                with_attributes(
                    {
                        "key": "gen_ai.input.messages",
                        "value": {"stringValue": '[{"role":"user"}]'},
                    }
                )
            ),
        ]
        for data in cases:
            with self.subTest(data):
                with self.assertRaises(ValueError):
                    module.check(data)

    def test_metrics(self):
        point = {
            "timeUnixNano": "1",
            "asInt": "-5",
            "attributes": [{"key": "k", "value": {"boolValue": True}}],
            "exemplars": [{"spanId": "eee19b7ec3c1b174", "asDouble": Number("1.5")}],
        }
        histogram = {
            "count": "2",
            "bucketCounts": ["1", Number("1")],
            "explicitBounds": [Number("0.5")],
        }
        metric = {
            "name": "m",
            "sum": {"dataPoints": [point], "aggregationTemporality": Number("2")},
            "histogram": {"dataPoints": [histogram]},
        }
        module.check({"resourceMetrics": [{"scopeMetrics": [{"metrics": [metric]}]}]})
        cases = [
            dict(metric, sum={"dataPoints": [dict(point, asInt="1.5")]}),
            dict(metric, histogram={"dataPoints": [{"bucketCounts": [None]}]}),
            dict(metric, histogram={"dataPoints": [{"bucketCounts": ["-1"]}]}),
            dict(metric, sum={"isMonotonic": "true"}),
        ]
        for case in cases:
            data = {"resourceMetrics": [{"scopeMetrics": [{"metrics": [case]}]}]}
            with self.subTest(case):
                with self.assertRaises(ValueError):
                    module.check(data)

    def test_repeated_keys(self):
        pair = {"key": "k", "value": {"stringValue": "v"}}
        with self.assertRaises(ValueError):
            module.check(traces(with_attributes(pair, pair)))
        kvlist = {"kvlistValue": {"values": [pair, pair]}}
        with self.assertRaises(ValueError):
            module.check(traces(with_attributes({"key": "m", "value": kvlist})))

    def test_two_kinds_in_a_line(self):
        with self.assertRaises(ValueError):
            module.check({"resourceSpans": [], "resourceMetrics": []})

    def test_the_eight_on_a_scope_are_not_checked_as_sets(self):
        pair = {"key": "gen_ai.input.messages", "value": {"stringValue": "x"}}
        module.check(traces(SPAN, {"attributes": [pair]}))


class TestKindOfData(unittest.TestCase):
    def test_kind_of_data(self):
        cases = [
            (traces(SPAN), "resourceSpans"),
            ({"resourceLogs": []}, "resourceLogs"),
            ({"resourceMetrics": None}, ""),
            ({}, ""),
        ]
        for data, expected in cases:
            with self.subTest(data):
                self.assertEqual(module.kind_of_data(data), expected)


class TestLeftBehind(unittest.TestCase):
    def test_names(self):
        profiling = {"stringValueStrindex": Number("2")}
        span = with_attributes(
            {"key": "k", "keyStrindex": Number("1"), "value": profiling},
            {"key": "m", "value": {"stringValue": "v", "other": 1}},
        )
        data = traces(dict(span, extra=True))
        self.assertEqual(
            module.left_behind(data),
            {"extra", "keyStrindex", "stringValueStrindex", "other"},
        )
        self.assertEqual(
            module.left_behind({"resourceMetrics": []}), {"resourceMetrics"}
        )

    def test_nothing(self):
        self.assertEqual(module.left_behind(traces(SPAN)), set())


class TestAttributeValues(unittest.TestCase):
    def test_values(self):
        cases = [
            ({"stringValue": "x"}, [["", "string", "x"]]),
            ({"boolValue": False}, [["", "boolean", "false"]]),
            ({"intValue": "5"}, [["", "number", "5"]]),
            ({"doubleValue": Number("0.5")}, [["", "number", "0.5"]]),
            ({"doubleValue": "NaN"}, [["", "string", "NaN"]]),
            ({"bytesValue": "aGk="}, [["", "string", "aGk="]]),
            ({}, [["", "null", ""]]),
            ({"stringValueStrindex": Number("3")}, [["", "null", ""]]),
            (
                {"arrayValue": {"values": [{"stringValue": "p"}]}},
                [["", "array", ""], ["/0", "string", "p"]],
            ),
            (
                {"kvlistValue": {"values": [{"key": "k", "value": {"intValue": "1"}}]}},
                [["", "object", ""], ["/k", "number", "1"]],
            ),
            ({"intValue": "1", "stringValue": "last"}, [["", "string", "last"]]),
        ]
        for value, expected in cases:
            with self.subTest(value):
                self.assertEqual(attribute_rows(value), expected)


class TestEntries(unittest.TestCase):
    def test_the_eight_on_a_span(self):
        span = with_attributes(
            {"key": "gen_ai.request.model", "value": {"stringValue": "m"}},
            {
                "key": "gen_ai.system_instructions",
                "value": {"stringValue": '[{"type":"text","content":"Be brief."}]'},
            },
        )
        a = "/1/resourceSpans/0/scopeSpans/0/spans/0"
        found = module.rows(1, traces(span))
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
        found = module.rows(0, traces(SPAN, {"attributes": [pair]}))
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
        found = module.rows(0, traces(SPAN))
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
                A,
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

    def test_absent_or_null_resource_has_no_row(self):
        for resource in ("absent", None):
            with self.subTest(resource):
                data = traces(SPAN)
                if resource is None:
                    data["resourceSpans"][0]["resource"] = None
                else:
                    del data["resourceSpans"][0]["resource"]
                names = [name for name, _ in module.rows(0, data)]
                self.assertNotIn("resourceSpans.resource", names)

    def test_body(self):
        pair = {"key": "msg", "value": {"stringValue": "hi"}}
        record = {"body": {"kvlistValue": {"values": [pair]}}}
        data = {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}
        a = "/0/resourceLogs/0/scopeLogs/0/logRecords/0"
        self.assertEqual(
            [e for e in module.rows(0, data) if e[0].endswith(".body")],
            [
                ("resourceLogs.scopeLogs.logRecords.body", [a, "", "object", ""]),
                ("resourceLogs.scopeLogs.logRecords.body", [a, "/msg", "string", "hi"]),
            ],
        )

    def test_null_body_has_no_rows(self):
        data = {"resourceLogs": [{"scopeLogs": [{"logRecords": [{"body": None}]}]}]}
        found = module.rows(0, data)
        self.assertFalse([e for e in found if e[0].endswith(".body")])

    def test_metrics_yield_no_rows(self):
        self.assertEqual(module.rows(0, decode('{"resourceMetrics":[{}]}')), [])


if __name__ == "__main__":
    unittest.main()
