"""Tests of integrations: the integration that reads or installs."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from living_memory import integrations as module

FILE = types.SimpleNamespace(EXTENSION=".x")
DIRECTORY = types.SimpleNamespace(FILE="index.x")
HOST = types.SimpleNamespace(HOST="host")


class TestReader(unittest.TestCase):
    # A file is read by the integration of its extension, a directory by
    # that of the file it holds.
    def test_file_and_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "index.x").write_bytes(b"")
            with mock.patch.object(module, "_members", return_value=[FILE, DIRECTORY]):
                self.assertIs(module.reader(Path(folder, "a.x")), FILE)
                self.assertIs(module.reader(Path(folder)), DIRECTORY)
                with self.assertRaises(LookupError):
                    module.reader(Path(folder, "a.y"))


class TestInstaller(unittest.TestCase):
    def test_host(self):
        with mock.patch.object(module, "_members", return_value=[FILE, HOST]):
            self.assertIs(module.installer("host"), HOST)
            with self.assertRaises(LookupError):
                module.installer("other")


if __name__ == "__main__":
    unittest.main()
