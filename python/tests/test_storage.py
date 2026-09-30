"""Tests of _storage: what a data bank holds of an input."""

import unittest

from living_memory import _storage as module

ROOT = {
    "sheet name": "t",
    "header": ["pointer", "n"],
    "records": [["/0", "1"], ["/1", "2"]],
}
RUNS = {
    "sheet name": "runs",
    "header": ["pointer", "page", "line", "position", "value"],
    "records": [["/0/s", "0", "0", "0", "a"], ["/2/s", "0", "0", "0", "b"]],
}


class TestCount(unittest.TestCase):
    # storage.3, storage.5, key.2: the values stored are the records of
    # the sheet of the input values that are a value's own; none without
    # it.
    def test_count(self):
        self.assertEqual(module.count([(0, ROOT), (3, RUNS)]), 2)
        self.assertEqual(module.count([(3, RUNS)]), 0)
        molten = {**ROOT, "records": [["/0", "a"], ["/0/0", "b"], ["/1", "c"]]}
        self.assertEqual(module.count([(0, molten)]), 2)


class TestWhole(unittest.TestCase):
    # storage.4: each sheet keeps only the records of the values stored
    # whole.
    def test_whole(self):
        kept = module.whole([(0, ROOT), (3, RUNS)], 2)
        self.assertEqual(kept[1][1]["records"], [["/0/s", "0", "0", "0", "a"]])
        self.assertEqual(kept[0][1]["records"], ROOT["records"])


if __name__ == "__main__":
    unittest.main()
