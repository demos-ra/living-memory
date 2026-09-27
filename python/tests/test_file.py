"""Tests of _file: the whole input as one MTSV file."""

import unittest

from living_memory import _file as module
from living_memory import _key
from living_memory._relation import Layout, Placed, Relation, Segment


class TestWrite(unittest.TestCase):
    # file.2, file.3: every sheet that holds a record, an FF line
    # before each; a sheet that holds none is left out.
    def test_sheets_with_records(self):
        below = Relation(Segment("property", (), "a"), _key.SUBORDINATE, (), ())
        root = Relation(Segment("root", (), "t"), _key.ROOT, (), (below,))
        placed = [Placed(root, {"pointer": "/0"}, {})]
        text = module.write(Layout(root, None, {}, ()), placed)
        self.assertEqual(text, "\ft\npointer\n/0\n")

    # value.2, order.1: every sheet the schema gives, in the file's
    # order, those with no record included.
    def test_names(self):
        below = Relation(Segment("property", (), "a"), _key.SUBORDINATE, (), ())
        root = Relation(Segment("root", (), "t"), _key.ROOT, (), (below,))
        self.assertEqual(module.names(Layout(root, None, {}, ())), ["t", "t.a"])


if __name__ == "__main__":
    unittest.main()
