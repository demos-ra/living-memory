"""Tests of _storage: what a data bank holds of an input."""

import unittest

from living_memory import _storage as module

ROOT = {
    "sheet name": "t",
    "header": ["_input value", "n"],
    "records": [["0", "1"], ["1", "2"]],
}
RUNS = {
    "sheet name": "_runs",
    "header": ["_input value", "_instance", "_page", "_line", "_position", "_value"],
    "records": [["0", "1", "0", "0", "0", "a"], ["2", "1", "0", "0", "0", "b"]],
}


class TestPosition(unittest.TestCase):
    # key.2, key.5: a record's value is at its _input value.
    def test_position(self):
        self.assertEqual(module.position(RUNS, RUNS["records"][1]), 2)


class TestWhole(unittest.TestCase):
    # storage.4: each sheet keeps only the records of the values stored
    # whole.
    def test_whole(self):
        kept = module.whole([(0, ROOT), (3, RUNS)], 2)
        self.assertEqual(kept[1][1]["records"], [RUNS["records"][0]])
        self.assertEqual(kept[0][1]["records"], ROOT["records"])


if __name__ == "__main__":
    unittest.main()
