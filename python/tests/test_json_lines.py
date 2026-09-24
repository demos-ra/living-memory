"""Tests of _json_lines: how a JSON Lines file is split into lines."""

import unittest

from living_memory import _json_lines


class TestLines(unittest.TestCase):
    def test_text_and_bytes(self):
        cases = [
            ("", []),
            ("a", ["a"]),
            ("a\n", ["a"]),
            ("a\nb\n", ["a", "b"]),
            ("a\r\nb", ["a\r", "b"]),
            ("a\n\n", ["a", ""]),
        ]
        for document, expected in cases:
            with self.subTest(document):
                self.assertEqual(_json_lines.lines(document), expected)
                self.assertEqual(
                    _json_lines.lines(document.encode()),
                    [line.encode() for line in expected],
                )

    def test_byte_order_mark_refused(self):
        for document in ("﻿{}\n", "﻿{}\n".encode()):
            with self.subTest(document):
                with self.assertRaises(ValueError):
                    _json_lines.lines(document)


if __name__ == "__main__":
    unittest.main()
