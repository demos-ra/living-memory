"""Tests of _command: a source read by its integration, as MTSV."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from living_memory import _command as module
from living_memory import integrations

SCHEMA = b'{"title":"t","type":"object","required":["n"],"properties":'
SCHEMA += b'{"n":{"type":"number"}},"additionalProperties":false}'
READER = types.SimpleNamespace(
    schema=lambda: SCHEMA,
    values=lambda path, held: [] if held else [b'{"n":1}', b'{"n":2}'],
)


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            module.main(list(argv))
            code = 0
        except SystemExit as exit:
            code = exit.code if isinstance(exit.code, int) else 1
            err.write(exit.code if isinstance(exit.code, str) else "")
    return code, out.getvalue(), err.getvalue()


class TestConvert(unittest.TestCase):
    # The output is the input's name with .mtsv, beside it, unless -o or
    # an output operand names it; it is kept as its sheets' files, and
    # a second conversion appends only what the reader gives after the
    # output's last record.
    def test_output_beside_the_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "calls.x")
            with mock.patch.object(integrations, "reader", return_value=READER):
                self.assertEqual(run(str(source))[0], 0)
                self.assertEqual(run(str(source))[0], 0)
                self.assertEqual(run(str(source), "-o", f"{folder}/o.mtsv")[0], 0)
            written = Path(folder, "calls.mtsv", "0 t.mtsv").read_text("utf-8")
            self.assertEqual(written, "\ft\npointer\tn\n/0\t1\n/1\t2\n")
            other = Path(folder, "o.mtsv", "0 t.mtsv").read_text("utf-8")
            self.assertEqual(other, written)

    # POSIX.1-2017 XBD 12.2, Guideline 13: "-" is standard output.
    def test_standard_output(self):
        with mock.patch.object(integrations, "reader", return_value=READER):
            with mock.patch.object(module.sys, "stdout") as stdout:
                module.main(["calls.x", "-"])
        stdout.buffer.write.assert_called_once()

    # GNU Coding Standards 4.4: a usage error reads "PROGRAM: MESSAGE".
    def test_usage_errors(self):
        for argv in (["missing.none"], ["-"], [], ["a.x", "b", "-o", "c"]):
            with self.subTest(argv=argv):
                code, _, err = run(*argv)
                self.assertEqual(code, 2)
                self.assertIn("living-memory: ", err)

    # An input that does not conform is refused by its place.
    def test_non_conforming(self):
        reader = types.SimpleNamespace(
            schema=lambda: SCHEMA, values=lambda path, held: [b"{}"]
        )
        with mock.patch.object(integrations, "reader", return_value=reader):
            code, _, err = run("calls.x", "-")
        self.assertEqual(code, 1)
        self.assertIn('living-memory: calls.x: an assertion fails: value 0, ""', err)


class TestOptions(unittest.TestCase):
    # GNU Coding Standards 4.8.1: --version names the program and its
    # version.
    def test_version(self):
        code, out, _ = run("--version")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("living-memory "))

    # Before changing another program's configuration, the command asks,
    # and refuses without a terminal.
    def test_install_needs_a_terminal(self):
        host = types.SimpleNamespace(HOST="h", change=str, install=mock.Mock())
        with mock.patch.object(integrations, "installer", return_value=host):
            with mock.patch.object(module.sys.stdin, "isatty", return_value=False):
                code, _, err = run("--install=h")
        self.assertEqual(code, 1)
        self.assertIn("needs a terminal", err)
        host.install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
