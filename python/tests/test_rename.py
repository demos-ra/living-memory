"""Tests of _rename: a file replaced whole."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _rename as module


class TestReplace(unittest.TestCase):
    # The file takes the new bytes whole, its folder made where missing,
    # and nothing else is left beside it.
    def test_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "a", "s.json")
            module.replace(path, b"1")
            module.replace(path, b"2")
            self.assertEqual(path.read_bytes(), b"2")
            self.assertEqual([p.name for p in path.parent.iterdir()], ["s.json"])


if __name__ == "__main__":
    unittest.main()
