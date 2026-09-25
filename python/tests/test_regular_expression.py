"""Tests of _regular_expression: the subset JSON Schema recommends."""

import unittest

from living_memory import _regular_expression as module


class TestSearch(unittest.TestCase):
    # value.3: a pattern of the subset matches, not anchored.
    def test_subset_matches_unanchored(self):
        cases = [
            ("es", "expression", True),
            ("^x-", "x-a", True),
            ("^x-", "a-x-", False),
            ("a$", "a\n", False),
            ("a$", "ba", True),
            ("[a-c]+", "zzb", True),
            ("[^a]", "a", False),
            ("[a-]", "-", True),
            ("(ab|cd){2}", "abcd", True),
            ("(ab|cd){3}", "abcd", False),
            ("a{1,}?b", "aab", True),
            ("^a{2,3}$", "aaaa", False),
            ("^a{2}$", "aa", True),
            ("^(a|)*$", "aaa", True),
            ("^b?c*$", "", True),
            ("é", "café", True),
        ]
        for pattern, text, expected in cases:
            with self.subTest(pattern=pattern, text=text):
                self.assertEqual(module.search(pattern, text), expected)

    # module.1: any other token makes the schema non-conforming.
    def test_tokens_outside_the_subset(self):
        for pattern in ("\\d", ".", "(?:a)", "[a\\]]", "[]", "a)", "(a", "*", "{"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    module.compile(pattern)
        for pattern in ("a**", "^*", "a{2,1}", "[z-a]", "a{x}"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    module.compile(pattern)


if __name__ == "__main__":
    unittest.main()
