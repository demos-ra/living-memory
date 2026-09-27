"""Tests of _order: the order of sheets, records and columns."""

import unittest

from living_memory import _order as module

TREE = {"t": ["a", "s", "b"], "a": ["a1", "s"], "a1": [], "b": ["a"], "s": []}


class TestSheets(unittest.TestCase):
    # order.1: each sheet directly followed by its subordinate sheets,
    # each once, in the place of its first reference, each given with
    # the sheets above it; the shared sheets last.
    def test_preorder(self):
        found = module.sheets("t", TREE.__getitem__, ["s"])
        expected = [("t",), ("t", "a"), ("t", "a", "a1"), ("t", "b")]
        expected += [("t", "a", "s")]
        self.assertEqual(found, expected)


class TestRecords(unittest.TestCase):
    # order.2: records in the order placed, never sorted.
    def test_grouped_in_order(self):
        placed = [("a", 2), ("t", 1), ("a", 0)]
        found = module.records(["t", "a"], placed, lambda one: one[0])
        self.assertEqual(found, [[("t", 1)], [("a", 2), ("a", 0)]])

    # order.2: members as written, an instance before those within it.
    def test_members_and_instances(self):
        value = {"b": [1], "a": 2}
        self.assertEqual(module.members(value), [("b", [1]), ("a", 2)])
        self.assertEqual(module.members([5, 6]), [(0, 5), (1, 6)])
        self.assertEqual(
            module.instances(value),
            [((), value), (("b",), [1]), (("b", 0), 1), (("a",), 2)],
        )


class TestDepth(unittest.TestCase):
    # value.3: the instances of a value of any depth are walked.
    def test_any_depth(self):
        value: list = []
        for _ in range(5000):
            value = [value]
        self.assertEqual(len(module.instances(value)), 5001)


class TestColumns(unittest.TestCase):
    # order.3: the key columns first, then the simple domains.
    def test_keys_first(self):
        self.assertEqual(module.columns(("pointer",), ("n",)), ["pointer", "n"])


if __name__ == "__main__":
    unittest.main()
