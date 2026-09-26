"""Tests of _value: how each input value is read, or rejected."""

import unittest

from living_memory import _value as module
from living_memory._json import decode
from living_memory._json_pointer import PlacedError

ROOT = decode(
    b'{"title":"t","type":"object","required":["p"],"properties":'
    b'{"p":{"type":"object","properties":{"k":{"type":"number"}}}}}'
)


class TestRead(unittest.TestCase):
    # value.2: members in the order written, numbers as written.
    def test_read(self):
        value = module.read(b'{"p":{"k":1.50},"a":1}', ROOT)
        self.assertEqual(list(value), ["p", "a"])
        self.assertEqual(value["p"]["k"], "1.50")

    # value.3: a value is rejected at the deepest place that fails.
    def test_rejected(self):
        cases = [
            (b'{"p":', ""),
            (b'{"p":{},"p":{}}', ""),
            (b'{"p":{"k":"x"}}', "/p/k"),
            (b"{}", ""),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                with self.assertRaises(PlacedError) as raised:
                    module.read(data, ROOT)
                self.assertEqual(raised.exception.pointer, expected)


if __name__ == "__main__":
    unittest.main()
