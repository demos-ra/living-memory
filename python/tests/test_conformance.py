"""Run every conformance case against the living_memory package."""

import logging
import unittest

import mtsv
from living_memory.integrations import otlp_json

from support import CONFORMANCE, cases


class TestConformance(unittest.TestCase):
    def setUp(self):
        # The cases leave things behind by design; the sheets are what
        # is compared, so the warnings are not shown.
        logger = logging.getLogger("living_memory")
        self.addCleanup(setattr, logger, "disabled", logger.disabled)
        logger.disabled = True

    def test_converted(self):
        for folder in ("conforming", "cannot-be-represented"):
            for path in cases(folder):
                with self.subTest(str(path.relative_to(CONFORMANCE))):
                    with path.open("rb") as file:
                        actual = otlp_json.load(file)
                    with path.with_suffix(".mtsv").open("rb") as file:
                        expected = mtsv.load(file)
                    self.assertEqual(actual, expected)

    def test_rejected(self):
        for path in cases("non-conforming"):
            with self.subTest(str(path.relative_to(CONFORMANCE))):
                with path.open("rb") as file:
                    with self.assertRaises(otlp_json.OTLPDecodeError):
                        otlp_json.load(file)


if __name__ == "__main__":
    unittest.main()
