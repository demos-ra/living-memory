"""Tests of __init__: the public interface."""

import io
import unittest

import living_memory
from living_memory.integrations import otlp_json

LINE = '{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"eventName":"e"}]}]}]}\n'


class TestInterface(unittest.TestCase):
    def test_loads(self):
        self.assertEqual(living_memory.loads(LINE), otlp_json.loads(LINE))

    def test_load(self):
        data = io.BytesIO(LINE.encode("utf-8"))
        self.assertEqual(living_memory.load(data), otlp_json.loads(LINE))

    def test_error_class(self):
        self.assertIs(living_memory.OTLPDecodeError, otlp_json.OTLPDecodeError)
        with self.assertRaises(living_memory.OTLPDecodeError):
            living_memory.loads("x\n")


if __name__ == "__main__":
    unittest.main()
