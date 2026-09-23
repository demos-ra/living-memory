"""Tests of integrations/providers: which provider formats exist."""

import unittest

from living_memory.integrations import providers


class TestProviders(unittest.TestCase):
    """No provider format exists yet."""

    def test_none(self):
        self.assertEqual(providers.PROVIDERS, {})

    def test_lookup(self):
        with self.assertRaises(LookupError):
            providers.lookup(".x")


if __name__ == "__main__":
    unittest.main()
