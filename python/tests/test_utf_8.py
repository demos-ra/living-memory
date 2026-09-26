"""Tests of _utf_8: how an octet sequence is read as UTF-8."""

import unittest

from living_memory import _utf_8 as module


class TestDecode(unittest.TestCase):
    # value.1: each value is a JSON text in UTF-8.
    def test_sequences_of_every_length(self):
        self.assertEqual(module.decode("aé€𝄞".encode()), "aé€𝄞")

    def test_not_utf_8(self):
        cases = [
            b"\xff",
            b"\xc0\x80",
            b"\xed\xa0\x80",
            b"\xf4\x90\x80\x80",
            b"\xe2\x82",
            b"\x80",
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    module.decode(data)


if __name__ == "__main__":
    unittest.main()
