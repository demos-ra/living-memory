"""Tests of _store: the data bank's storage, each input's sheets kept as
files, each only appended to."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _store as module

NAMES = ["t", "t.a"]
PART = "\ft\npointer\n/1\n\ft.a\nparent\tpointer\n/1\t/1/a/0\n"


class TestAppend(unittest.TestCase):
    # storage.3, storage.4: each sheet's file is named by its place and
    # its name; its FF line and header are written once, and records are
    # only appended.
    def test_append(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            module.append(store, "\ft\npointer\n/0\n", NAMES)
            module.append(store, PART, NAMES)
            self.assertEqual((store / "0 t.mtsv").read_text(), "\ft\npointer\n/0\n/1\n")
            self.assertEqual(
                (store / "1 t.a.mtsv").read_text(),
                "\ft.a\nparent\tpointer\n/1\t/1/a/0\n",
            )


class TestRepair(unittest.TestCase):
    # storage.4: a line not ended, and a record of a value the sheet of
    # the input values does not hold, are no part of the storage.
    def test_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            (store / "0 t.mtsv").write_text("\ft\npointer\n/0\n/1")
            (store / "1 t.a.mtsv").write_text(
                "\ft.a\nparent\tpointer\n/0\t/0/a/0\n/1\t/1/a/0\n"
            )
            module.repair(store, NAMES)
            self.assertEqual((store / "0 t.mtsv").read_text(), "\ft\npointer\n/0\n")
            self.assertEqual(
                (store / "1 t.a.mtsv").read_text(),
                "\ft.a\nparent\tpointer\n/0\t/0/a/0\n",
            )

    # A file left with less than its FF line and header is left empty.
    def test_header_not_ended(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            (store / "0 t.mtsv").write_text("\ft\npoin")
            module.repair(store, NAMES)
            self.assertEqual((store / "0 t.mtsv").read_text(), "")


class TestStored(unittest.TestCase):
    # storage.3: an input's stored sheets, each with its place among
    # every sheet the schema gives; none where it holds none.
    def test_stored(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual(module.stored(store, NAMES), [])
            module.append(store, "\ft.a\nparent\tpointer\n/0\t/0/a/0\n", NAMES)
            found = module.stored(store, NAMES)
            self.assertEqual([place for place, _ in found], [1])
            self.assertEqual(found[0][1]["records"], [["/0", "/0/a/0"]])


class TestCommunicated(unittest.TestCase):
    # communication.6: how many values have been communicated: none at
    # first, then as marked, kept apart from the lines read.
    def test_communicated(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual(module.communicated(store), 0)
            module.mark_communicated(store, 3)
            module.mark_read(store, 7)
            self.assertEqual((module.communicated(store), module.read(store)), (3, 7))


class TestInputs(unittest.TestCase):
    # storage.2, communication.3: the inputs stored, in the order first
    # stored, each once.
    def test_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "o.mtsv")
            store.mkdir()
            self.assertEqual(module.inputs(store), [])
            for name in ("d2/b", "d1/a", "d2/b"):
                module.add_input(store, name)
            self.assertEqual(module.inputs(store), ["d2/b", "d1/a"])


class TestLocked(unittest.TestCase):
    # The storage's folder is made, and its lock held, for a conversion.
    def test_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "o.mtsv")
            with module.locked(store):
                self.assertTrue(store.is_dir())


if __name__ == "__main__":
    unittest.main()
