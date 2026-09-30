"""Tests of _communication: the names of what a data bank stores, and
its records by their keys."""

import unittest

import mtsv

from living_memory import _communication as module

ROOT = {
    "sheet name": "t",
    "header": ["_input value", "n"],
    "records": [["0", "1"], ["1", "2"]],
}
KIDS = {
    "sheet name": "t.k",
    "header": ["_input value", "_instance", "_parent", "_pointer", "_value"],
    "records": [["0", "2", "0", "/k/0", "5"], ["1", "2", "0", "/k/0", "6"]],
}
STORED = [(0, ROOT), (2, KIDS)]


class TestNames(unittest.TestCase):
    # communication.3: the inputs with the number of their values, the
    # sheets by place and name, and the fields by position, from 0,
    # each sheet and column named as the specification fixes it.
    def test_names(self):
        found = {
            s["sheet name"]: s for s in mtsv.loads(module.names([("i", STORED, 2)]))
        }
        self.assertEqual(list(found), ["_inputs", "_sheets", "_fields"])
        self.assertEqual(found["_inputs"]["header"], ["_input", "_values"])
        self.assertEqual(found["_inputs"]["records"], [["i", "2"]])
        self.assertEqual(
            found["_sheets"]["records"], [["i", "0", "t"], ["i", "2", "t.k"]]
        )
        self.assertEqual(
            found["_fields"]["header"], ["_input", "_place", "_position", "_field name"]
        )
        self.assertEqual(
            found["_fields"]["records"][2], ["i", "2", "0", "_input value"]
        )


class TestRecords(unittest.TestCase):
    # communication.4: the records of a range of values, both ends
    # included, of the places asked for, each in its stored sheet; a
    # sheet holding none is left out.
    def test_records(self):
        both = mtsv.loads(module.records(STORED, 1, 1))
        self.assertEqual(
            [s["records"] for s in both],
            [[["1", "2"]], [["1", "2", "0", "/k/0", "6"]]],
        )
        kids = mtsv.loads(module.records(STORED, 0, 1, [2]))
        self.assertEqual([s["sheet name"] for s in kids], ["t.k"])
        self.assertEqual(module.records(STORED, 5, 9), "")
        self.assertEqual(module.records(STORED, 0, 1, []), "")


if __name__ == "__main__":
    unittest.main()
