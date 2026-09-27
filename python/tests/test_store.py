"""Tests of _store: the output kept as its sheets' files."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _store as module

NAMES = ["t", "t.a"]
PART = "\ft\npointer\n/1\n\ft.a\nparent\tpointer\n/1\t/1/a/0\n"


class TestAppend(unittest.TestCase):
    # Each sheet's file is named by its place and its name; a header is
    # written once, and records are only appended.
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
    # A line not ended, and a record of a value the sheet of the input
    # values does not hold, are left out.
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


class TestHeld(unittest.TestCase):
    # The records of the sheet of the input values, by the header's
    # names; none where the output holds none.
    def test_held(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual(module.held(store, NAMES), [])
            (store / "0 t.mtsv").write_text("\ft\npointer\n/0\n")
            self.assertEqual(module.held(store, NAMES), [{"pointer": "/0"}])


class TestView(unittest.TestCase):
    # Each sheet file in its order, with its name, header and the
    # position each record comes from; the folder made on the first
    # append.
    def test_view(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "d", "s")
            self.assertEqual(module.view(store), [])
            module.append(store, "\ft\npointer\n/0\n", NAMES)
            module.append(store, PART, NAMES)
            found = module.view(store)
            self.assertEqual(
                [(f["file"], f["name"], f["positions"]) for f in found],
                [("0 t.mtsv", "t", [0, 1]), ("1 t.a.mtsv", "t.a", [1])],
            )
            self.assertEqual(found[1]["header"], ["parent", "pointer"])


class TestGiven(unittest.TestCase):
    # How many input values a context has given: none at first, then as
    # marked.
    def test_given(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual(module.given(store), 0)
            module.mark_given(store, 3)
            self.assertEqual(module.given(store), 3)


class TestRead(unittest.TestCase):
    # How many lines of the source have been read: none at first, then
    # as marked, kept apart from what a context has given.
    def test_read(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            self.assertEqual(module.read(store), 0)
            module.mark_read(store, 7)
            module.mark_given(store, 3)
            self.assertEqual((module.read(store), module.given(store)), (7, 3))


class TestInputs(unittest.TestCase):
    # The inputs an output holds, each by the path of its folder.
    def test_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "o.mtsv")
            self.assertEqual(module.inputs(store), [])
            for name in ("d2/b", "d1/a"):
                module.append(store / name, "\ft\npointer\n/0\n", NAMES)
            self.assertEqual(module.inputs(store), ["d1/a", "d2/b"])


class TestLocked(unittest.TestCase):
    # The output's folder is made, and its lock held, for a conversion.
    def test_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "o.mtsv")
            with module.locked(store):
                self.assertTrue(store.is_dir())


if __name__ == "__main__":
    unittest.main()
