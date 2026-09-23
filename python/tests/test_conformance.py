"""Run every conformance case against the living_memory package."""

import unittest

import mtsv
from living_memory.integrations import otlp_json

from support import CONFORMANCE, cases


class TestConformance(unittest.TestCase):
    """Each case's input converts to the sheets of its expected file."""

    def test_cases(self):
        for path in cases():
            with self.subTest(str(path.relative_to(CONFORMANCE))):
                with path.open("rb") as file:
                    actual = otlp_json.load(file)
                with path.with_suffix(".mtsv").open("rb") as file:
                    expected = mtsv.load(file)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
