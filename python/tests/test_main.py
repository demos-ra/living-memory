"""Tests of __main__: the command's entry point."""

import unittest
from unittest import mock

from living_memory import __main__


class TestMain(unittest.TestCase):
    def test_main(self):
        with mock.patch("living_memory._command.run") as run:
            __main__.main(["in.jsonl"])
        run.assert_called_once_with(["in.jsonl"])


if __name__ == "__main__":
    unittest.main()
