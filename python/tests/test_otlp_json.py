"""Tests of integrations/otlp_json: how an OTLP file is read."""

import io
import unittest

from living_memory.integrations import otlp_json

SPANS = (
    '{"resourceSpans":[{"scopeSpans":[{"spans":[{'
    '"traceId":"5b8efff798038103d269b633813fc60c","spanId":"eee19b7ec3c1b174",'
    '"name":"n"}]}]}]}'
)


def rows(sheets, name):
    return next(s["records"] for s in sheets if s["sheet name"] == name)


class TestLoads(unittest.TestCase):
    def test_empty_file_has_every_sheet(self):
        sheets = otlp_json.loads("")
        self.assertEqual(len(sheets), 245)
        self.assertFalse(any(s["records"] for s in sheets))

    def test_line_index_in_address(self):
        sheets = otlp_json.loads("{}\n" + SPANS + "\n")
        self.assertEqual(
            rows(sheets, "resourceSpans.scopeSpans.spans")[0][:3],
            [
                "/1/resourceSpans/0/scopeSpans/0/spans/0",
                "5b8efff798038103d269b633813fc60c",
                "eee19b7ec3c1b174",
            ],
        )

    def test_final_line_feed_optional(self):
        self.assertEqual(otlp_json.loads(SPANS), otlp_json.loads(SPANS + "\n"))

    def test_crlf(self):
        self.assertEqual(otlp_json.loads(SPANS + "\r\n"), otlp_json.loads(SPANS + "\n"))

    def test_rejected_lines_are_counted_from_1(self):
        cases = [
            (SPANS + "\n\n" + SPANS + "\n", 2),
            ("x\n", 1),
            (SPANS + "\n[NaN]\n", 2),
            ("﻿" + SPANS + "\n", 1),
            (SPANS + "\n" + '{"resourceLogs":[]}' + "\n", 2),
            ('{"resourceSpans":[],"resourceLogs":[]}' + "\n", 1),
        ]
        for text, lineno in cases:
            with self.subTest(text):
                with self.assertRaises(otlp_json.OTLPDecodeError) as found:
                    otlp_json.loads(text)
                self.assertEqual(found.exception.lineno, lineno)
                self.assertIsInstance(found.exception, ValueError)

    def test_left_behind_is_a_warning(self):
        text = SPANS.replace('"name":"n"', '"name":"n\\tm","extra":1') + "\n"
        with self.assertLogs("living_memory", "WARNING") as logs:
            otlp_json.loads(text)
        self.assertEqual(
            logs.records[0].left_behind,
            ["extra", "resourceSpans.scopeSpans.spans.name"],
        )


class TestLoad(unittest.TestCase):
    def test_types(self):
        with self.assertRaises(TypeError):
            otlp_json.load(io.StringIO(SPANS))
        with self.assertRaises(TypeError):
            otlp_json.loads(SPANS.encode("utf-8"))

    def test_invalid_utf_8(self):
        data = SPANS.replace('"n"', '"\xff"').encode("latin-1")
        sheets = otlp_json.load(io.BytesIO(data))
        self.assertEqual(rows(sheets, "resourceSpans.scopeSpans.spans")[0][6], "�")


if __name__ == "__main__":
    unittest.main()
