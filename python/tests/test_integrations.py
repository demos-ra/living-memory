"""Tests of integrations/__init__: which reader reads which input."""

import io
import unittest

from living_memory import integrations
from living_memory.integrations import otlp_json


class TestLookup(unittest.TestCase):
    """lookup finds the reader of a file extension."""

    def test_jsonl(self):
        self.assertIs(integrations.lookup(".jsonl"), otlp_json)

    def test_unknown(self):
        with self.assertRaises(LookupError):
            integrations.lookup(".x")


class TestLoad(unittest.TestCase):
    """load reads with the reader of the extension."""

    def test_load(self):
        self.assertEqual(
            integrations.load(".jsonl", io.BytesIO(b"")), otlp_json.loads("")
        )


if __name__ == "__main__":
    unittest.main()
