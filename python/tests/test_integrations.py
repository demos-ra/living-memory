"""Tests of integrations/__init__: which reader reads which input."""

import io
import tempfile
import unittest
from pathlib import Path

from living_memory import integrations
from living_memory.integrations import otlp_json
from living_memory.integrations.providers.anthropic.claude_code import (
    install,
    raw_api_bodies,
)


class TestLookup(unittest.TestCase):
    def test_jsonl(self):
        self.assertIs(integrations.lookup(".jsonl"), otlp_json)

    def test_unknown(self):
        with self.assertRaises(LookupError):
            integrations.lookup(".x")

    def test_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(LookupError):
                integrations.lookup_directory(folder)
            (folder / "index.jsonl").write_text("", encoding="utf-8")
            self.assertIs(integrations.lookup_directory(folder), raw_api_bodies)

    def test_plugin(self):
        self.assertIs(integrations.lookup_plugin("claude-code"), install)


class TestLoad(unittest.TestCase):
    def test_load(self):
        self.assertEqual(
            integrations.load(".jsonl", io.BytesIO(b"")), otlp_json.loads("")
        )


if __name__ == "__main__":
    unittest.main()
