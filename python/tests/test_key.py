"""Tests of _key: each record's primary key."""

import unittest

from living_memory import _key as module


class TestColumns(unittest.TestCase):
    # key.2, key.3, key.5: the key columns of each kind of sheet.
    def test_kinds(self):
        self.assertEqual(module.ROOT, ("_input value",))
        self.assertEqual(
            module.SUBORDINATE, ("_input value", "_instance", "_parent", "_pointer")
        )
        self.assertEqual(
            module.RUN, ("_input value", "_instance", "_page", "_line", "_position")
        )


class TestInstances(unittest.TestCase):
    # key.3: an input value's instances, numbered from 0 in the order
    # written, the value itself first, by their pointers in the input.
    def test_numbered(self):
        value = {"b": [1], "a/c": 2}
        self.assertEqual(
            module.numbered(value, "/3"),
            {"/3": 0, "/3/b": 1, "/3/b/0": 2, "/3/a~1c": 3},
        )

    # key.2: an input value's pointer begins with its position.
    def test_position(self):
        self.assertEqual(module.root(3), "/3")
        for pointer in ("/12", "/12/a/0"):
            with self.subTest(pointer=pointer):
                self.assertEqual(module.position(pointer), 12)


class TestValues(unittest.TestCase):
    NUMBERS = {"/0": 0, "/0/a": 1, "/0/a/0": 2}

    # key.2, key.3: a record's keys, its parent's instance and the
    # pointer from it; a record without a parent leaves both empty.
    def test_records(self):
        self.assertEqual(
            module.values(module.ROOT, self.NUMBERS, "/0", None, ""),
            {"_input value": "0"},
        )
        self.assertEqual(
            module.values(module.SUBORDINATE, self.NUMBERS, "/0/a/0", "/0", "/a/0"),
            {"_input value": "0", "_instance": "2", "_parent": "0", "_pointer": "/a/0"},
        )
        self.assertEqual(
            module.values(module.SUBORDINATE, self.NUMBERS, "/0/a", None, ""),
            {"_input value": "0", "_instance": "1", "_parent": "", "_pointer": ""},
        )

    # key.5: a run is keyed by its string's input value and instance.
    def test_runs(self):
        self.assertEqual(
            module.run(self.NUMBERS, "/0/a", (1, 2, 3)),
            {
                "_input value": "0",
                "_instance": "1",
                "_page": "1",
                "_line": "2",
                "_position": "3",
            },
        )


if __name__ == "__main__":
    unittest.main()
