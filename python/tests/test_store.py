"""Tests of _store: the data bank's storage, each input's sheets kept as
files, each only appended to."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _store as module

NAMES = ["t", "t.a"]
K = "_input value\t_instance\t_parent\t_pointer"
PART = f"\ft\n_input value\n1\n\ft.a\n{K}\n1\t2\t0\t/a/0\n"


class TestAppend(unittest.TestCase):
    # storage.3, storage.4: each sheet's file is named by its place and
    # its name; its FF line and header are written once, and records are
    # only appended.
    def test_append(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            module.append(store, "\ft\n_input value\n0\n", NAMES)
            module.append(store, PART, NAMES)
            self.assertEqual(
                (store / "0 t.mtsv").read_text(), "\ft\n_input value\n0\n1\n"
            )
            self.assertEqual(
                (store / "1 t.a.mtsv").read_text(), f"\ft.a\n{K}\n1\t2\t0\t/a/0\n"
            )


class TestRepair(unittest.TestCase):
    # storage.4: a line not ended, and a record of a value not counted
    # as stored whole, are no part of the storage.
    def test_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            (store / "0 t.mtsv").write_text("\ft\n_input value\n0\n1\n")
            (store / "1 t.a.mtsv").write_text(
                f"\ft.a\n{K}\n0\t2\t0\t/a/0\n1\t2\t0\t/a/0"
            )
            module.mark_values(store, 1)
            module.repair(store, NAMES)
            self.assertEqual((store / "0 t.mtsv").read_text(), "\ft\n_input value\n0\n")
            self.assertEqual(
                (store / "1 t.a.mtsv").read_text(), f"\ft.a\n{K}\n0\t2\t0\t/a/0\n"
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
            module.append(store, f"\ft.a\n{K}\n0\t2\t0\t/a/0\n", NAMES)
            found = module.stored(store, NAMES)
            self.assertEqual([place for place, _ in found], [1])
            self.assertEqual(found[0][1]["records"], [["0", "2", "0", "/a/0"]])


class TestCounts(unittest.TestCase):
    # storage.4, storage.5, communication.6: how many values are stored
    # whole and how many communicated: none at first, then as marked,
    # each kept apart, and apart from the lines read.
    def test_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual((module.values(store), module.communicated(store)), (0, 0))
            module.mark_values(store, 5)
            module.mark_communicated(store, 3)
            module.mark_read(store, 7)
            found = (
                module.values(store),
                module.communicated(store),
                module.read(store),
            )
            self.assertEqual(found, (5, 3, 7))


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
