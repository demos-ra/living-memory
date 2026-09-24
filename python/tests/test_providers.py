"""Tests of integrations/providers: which providers exist."""

import tempfile
import unittest
from pathlib import Path

from living_memory.integrations import providers
from living_memory.integrations.providers import anthropic


class TestProviders(unittest.TestCase):
    def test_providers(self):
        self.assertEqual(providers.PROVIDERS, {"index.jsonl": "anthropic"})

    def test_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(LookupError):
                providers.lookup(folder)
            (folder / "index.jsonl").write_text("", encoding="utf-8")
            self.assertIs(providers.lookup(folder), anthropic)


if __name__ == "__main__":
    unittest.main()
