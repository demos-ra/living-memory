"""Tests of integrations/otlp_json: how an OTLP file is read."""

import io
import unittest

from living_memory.integrations import otlp_json

SPANS = (
    '{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"t","spanId":"s",'
    '"name":"n"}]}]}]}'
)


def rows(sheets, name):
    """Return the records of a named sheet."""
    return next(s["records"] for s in sheets if s["sheet name"] == name)


class TestLoads(unittest.TestCase):
    """loads reads each line, in order, keyed by its index."""

    def test_empty_file_has_every_sheet(self):
        sheets = otlp_json.loads("")
        self.assertEqual(len(sheets), 245)
        self.assertFalse(any(s["records"] for s in sheets))

    def test_line_index_in_address(self):
        sheets = otlp_json.loads('{"resourceMetrics":[]}\n' + SPANS + "\n")
        self.assertEqual(
            rows(sheets, "resourceSpans.scopeSpans.spans")[0][:3],
            ["/1/resourceSpans/0/scopeSpans/0/spans/0", "t", "s"],
        )

    def test_final_line_feed_optional(self):
        self.assertEqual(otlp_json.loads(SPANS), otlp_json.loads(SPANS + "\n"))

    def test_crlf(self):
        self.assertEqual(otlp_json.loads(SPANS + "\r\n"), otlp_json.loads(SPANS + "\n"))

    def test_blank_line_is_not_json(self):
        with self.assertRaisesRegex(ValueError, "^line 2: "):
            otlp_json.loads(SPANS + "\n\n" + SPANS + "\n")


class TestLoad(unittest.TestCase):
    """load decodes UTF-8, an invalid sequence as U+FFFD."""

    def test_invalid_utf_8(self):
        data = SPANS.replace('"n"', '"\xff"').encode("latin-1")
        sheets = otlp_json.load(io.BytesIO(data))
        self.assertEqual(rows(sheets, "resourceSpans.scopeSpans.spans")[0][6], "�")


if __name__ == "__main__":
    unittest.main()
