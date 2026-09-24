"""Tests of integrations/providers: which providers' products exist."""

import tempfile
import unittest
from pathlib import Path

from living_memory.integrations import providers
from living_memory.integrations.providers.anthropic.claude_code import (
    install,
    raw_api_bodies,
)


class TestProviders(unittest.TestCase):
    def test_providers(self):
        self.assertEqual(
            providers.PROVIDERS, {"index.jsonl": "anthropic.claude_code.raw_api_bodies"}
        )

    def test_plugins(self):
        self.assertEqual(
            providers.PLUGINS, {"claude-code": "anthropic.claude_code.install"}
        )

    def test_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(LookupError):
                providers.lookup(folder)
            (folder / "index.jsonl").write_text("", encoding="utf-8")
            self.assertIs(providers.lookup(folder), raw_api_bodies)

    def test_lookup_plugin(self):
        self.assertIs(providers.lookup_plugin("claude-code"), install)
        with self.assertRaises(LookupError):
            providers.lookup_plugin("x")


if __name__ == "__main__":
    unittest.main()
