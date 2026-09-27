"""Tests of _key: each record's primary key."""

import unittest

from living_memory import _key as module


class TestColumns(unittest.TestCase):
    # key.2, key.3, key.5: the key columns of each kind of sheet.
    def test_kinds(self):
        self.assertEqual(module.ROOT, ("pointer",))
        self.assertEqual(module.SUBORDINATE, ("parent", "pointer"))
        self.assertEqual(module.RUN, ("pointer", "page", "line", "position"))


class TestValues(unittest.TestCase):
    # key.2, key.4: an input value's pointer begins with its position,
    # and a member name is escaped.
    def test_pointers(self):
        self.assertEqual(module.root(3), "/3")
        self.assertEqual(module.member("/0", "a/b~c"), "/0/a~1b~0c")

    # key.3, key.5: the parent's key is copied down where the sheet has
    # it, and a run is keyed by its string's pointer.
    def test_records_and_runs(self):
        self.assertEqual(module.values(module.ROOT, None, "/0"), {"pointer": "/0"})
        self.assertEqual(
            module.values(module.SUBORDINATE, "/0", "/0/a"),
            {"parent": "/0", "pointer": "/0/a"},
        )
        self.assertEqual(
            module.run("/0/s", (1, 2, 3)),
            {"pointer": "/0/s", "page": "1", "line": "2", "position": "3"},
        )


if __name__ == "__main__":
    unittest.main()
