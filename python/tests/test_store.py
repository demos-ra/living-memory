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
    # values does not hold, are left out; that sheet's records are
    # returned.
    def test_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder)
            (store / "0 t.mtsv").write_text("\ft\npointer\n/0\n/1")
            (store / "1 t.a.mtsv").write_text(
                "\ft.a\nparent\tpointer\n/0\t/0/a/0\n/1\t/1/a/0\n"
            )
            self.assertEqual(module.repair(store, NAMES), [{"pointer": "/0"}])
            self.assertEqual((store / "0 t.mtsv").read_text(), "\ft\npointer\n/0\n")
            self.assertEqual(
                (store / "1 t.a.mtsv").read_text(),
                "\ft.a\nparent\tpointer\n/0\t/0/a/0\n",
            )

    def test_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(module.repair(Path(folder), NAMES), [])


class TestLocked(unittest.TestCase):
    # The output's folder is made, and its lock held, for a conversion.
    def test_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder, "o.mtsv")
            with module.locked(store):
                self.assertTrue(store.is_dir())


if __name__ == "__main__":
    unittest.main()
