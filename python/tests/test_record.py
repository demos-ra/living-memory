"""Tests of _record: each instance as one record."""

import unittest

from living_memory import _key
from living_memory import _record as module
from living_memory._json import Number
from living_memory._relation import Domain, Placed, Relation, Segment


class TestFields(unittest.TestCase):
    # record.1, field.1, field.2, relation.17: keys first, then each
    # column's text; a column written elsewhere is empty.
    def test_fields(self):
        kind, value = Domain("type", "type"), Domain("value", "value")
        absent = Domain("x", "value")
        one = Relation(Segment("root", (), "t"), _key.ROOT, (kind, value, absent), ())
        placed = Placed(one, {"pointer": "/0"}, {kind: Number("1"), value: Number("1")})
        self.assertEqual(module.fields(placed), ["/0", "number", "1", ""])

    # record.1: each run of text is one record.
    def test_run(self):
        run = Domain("value", "run")
        text = Relation(Segment("runs", (), "runs"), _key.RUN, (run,), ())
        keys = {"pointer": "/0/s", "page": "0", "line": "1", "position": "2"}
        self.assertEqual(
            module.fields(Placed(text, keys, {run: "a"})),
            ["/0/s", "0", "1", "2", "a"],
        )


if __name__ == "__main__":
    unittest.main()
