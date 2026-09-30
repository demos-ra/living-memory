"""Tests of _command: a source read by its integration, stored as MTSV
in a data bank, and communicated."""

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
    # The command's status, and what it writes to standard output, as
    # the bytes it writes, and to standard error.
    data = io.BytesIO()
    out, err = io.TextIOWrapper(data, "utf-8"), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            module.main(list(argv))
            code = 0
        except SystemExit as exit:
            code = exit.code if isinstance(exit.code, int) else 1
            err.write(exit.code if isinstance(exit.code, str) else "")
    out.flush()
    return code, data.getvalue().decode("utf-8"), err.getvalue()


class TestConvert(unittest.TestCase):
    # Register, Code row 104: the output is the input's name with .mtsv,
    # beside it, unless -o or an output operand names it; storage.2,
    # storage.5: the input is stored apart, named by its file, and a
    # second conversion inserts only what the reader gives after the
    # values stored.
    def test_output_beside_the_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "calls.x")
            with mock.patch.object(integrations, "reader", return_value=READER):
                self.assertEqual(run(str(source))[0], 0)
                self.assertEqual(run(str(source))[0], 0)
                self.assertEqual(run(str(source), "-o", f"{folder}/o.mtsv")[0], 0)
            written = Path(folder, "calls.mtsv", "calls", "0 t.mtsv").read_text("utf-8")
            self.assertEqual(written, "\ft\n_input value\tn\n0\t1\n1\t2\n")
            other = Path(folder, "o.mtsv", "calls", "0 t.mtsv").read_text("utf-8")
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


def several(folder: Path) -> types.SimpleNamespace:
    # A reader of two lines, each an input with one value, whose spent
    # file is the source file of its name.
    for name in ("a", "b"):
        Path(folder, name).write_text("x")
    return types.SimpleNamespace(
        schema=lambda: SCHEMA,
        lines=lambda path, start: [(1, "a"), (2, "b")][start:],
        inputs=lambda index, names: [f"d/{name}" for _, name in index],
        values=lambda path, name, held, index: [] if held else [b'{"n":1}'],
        read=lambda index, held, start: start + len(index),
        spent=lambda path, name, held, index: [Path(folder, name.split("/")[1])],
    )


class TestInputs(unittest.TestCase):
    # Each input is written to a folder of the output by its name, and
    # its spent files are removed once it is converted.
    def test_inputs_and_spent(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = several(Path(folder))
            with mock.patch.object(integrations, "reader", return_value=reader):
                self.assertEqual(run(f"{folder}/calls.x")[0], 0)
            for name in ("a", "b"):
                with self.subTest(name=name):
                    self.assertTrue(
                        Path(folder, "calls.mtsv", "d", name, "0 t.mtsv").exists()
                    )
                    self.assertFalse(Path(folder, name).exists())

    # raw_api_bodies.mtsv › values.6: how many lines are read is kept
    # beside the output, and the next conversion reads only those after.
    def test_lines_read(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = several(Path(folder))
            reader.lines = mock.Mock(side_effect=reader.lines)
            with mock.patch.object(integrations, "reader", return_value=reader):
                self.assertEqual(run(f"{folder}/calls.x")[0], 0)
                self.assertEqual(run(f"{folder}/calls.x")[0], 0)
            read = Path(folder, "calls.mtsv", ".read").read_text()
            self.assertEqual(read, "2\n")
            self.assertEqual(reader.lines.call_args.args[1], 2)

    # GNU Coding Standards 4.10, keep-files: -k keeps the spent files.
    def test_keep_files(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = several(Path(folder))
            with mock.patch.object(integrations, "reader", return_value=reader):
                self.assertEqual(run("-k", f"{folder}/calls.x")[0], 0)
            self.assertTrue(Path(folder, "a").exists())

    # --add-context=HOST converts, then writes the host's context, which
    # the host composes of the data bank's communications alone
    # (storage.6); what is new advances the number communicated
    # (communication.6).
    def test_add_context(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = several(Path(folder))
            seen = {}

            def context(hook_input, bank):
                seen["new"] = bank.new("d/a", None)
                return "\fmap\nx\n"

            host = types.SimpleNamespace(HOST="h", context=context)
            with mock.patch.object(integrations, "reader", return_value=reader):
                with mock.patch.object(integrations, "installer", return_value=host):
                    with mock.patch.object(module.sys, "stdin") as stdin:
                        stdin.buffer.read.return_value = b"{}"
                        with mock.patch.object(module.sys, "stdout") as stdout:
                            module.main(["--add-context=h", f"{folder}/calls.x"])
            stdout.buffer.write.assert_called_once_with(b"\fmap\nx\n")
            self.assertEqual(seen["new"], "\ft\n_input value\tn\n0\t1\n")
            given = Path(folder, "calls.mtsv", "d", "a", ".communicated").read_text()
            self.assertEqual(given, "1\n")


class TestCommunicate(unittest.TestCase):
    # communication.3: --names gives the names of what is stored.
    def test_names(self):
        with tempfile.TemporaryDirectory() as folder:
            source = f"{folder}/calls.x"
            with mock.patch.object(integrations, "reader", return_value=READER):
                run(source)
                code, out, _ = run("--names", source)
        self.assertEqual(code, 0)
        self.assertEqual(
            out,
            "\f_inputs\n_input\t_values\ncalls\t2\n"
            "\f_sheets\n_input\t_place\t_sheet name\ncalls\t0\tt\n"
            "\f_fields\n_input\t_place\t_position\t_field name\n"
            "calls\t0\t0\t_input value\ncalls\t0\t1\tn\n",
        )

    # communication.4: --filter gives one input's records of a range of
    # value positions, both ends included, of the places asked for.
    def test_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            source = f"{folder}/calls.x"
            with mock.patch.object(integrations, "reader", return_value=READER):
                run(source)
                whole = run("--filter=calls", source)[1]
                one = run("--filter=calls", "--values=1-1", "--places=0", source)[1]
                none = run("--filter=calls", "--places=1", source)[1]
        self.assertEqual(whole, "\ft\n_input value\tn\n0\t1\n1\t2\n")
        self.assertEqual(one, "\ft\n_input value\tn\n1\t2\n")
        self.assertEqual(none, "")

    # communication.5, communication.6: --new gives what is new, then
    # nothing, the number communicated kept.
    def test_new(self):
        with tempfile.TemporaryDirectory() as folder:
            source = f"{folder}/calls.x"
            with mock.patch.object(integrations, "reader", return_value=READER):
                run(source)
                first = run("--new=calls", source)[1]
                again = run("--new=calls", source)[1]
        self.assertEqual(first, "\ft\n_input value\tn\n0\t1\n1\t2\n")
        self.assertEqual(again, "")

    # A request names an input stored, and --values and --places narrow
    # only the requests they belong to.
    def test_request_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            source = f"{folder}/calls.x"
            with mock.patch.object(integrations, "reader", return_value=READER):
                run(source)
                self.assertEqual(run("--filter=none", source)[0], 1)
                for argv in (
                    ["--values=1", source],
                    ["--places=0", "--names", source],
                    ["--filter=calls", "--values=x", source],
                    ["--new=calls", "--places=a", source],
                ):
                    with self.subTest(argv=argv):
                        self.assertEqual(run(*argv)[0], 2)


class TestHook(unittest.TestCase):
    # install.mtsv › hooks.4: with --add-context a usage error exits
    # with status 1, never 2, which from a hook would erase the prompt.
    def test_status(self):
        code, _, err = run("--add-context=claude-code", "missing.none")
        self.assertEqual(code, 1)
        self.assertIn("living-memory: ", err)

    # Pandoc, Specifying formats: -f/--from names the reader, which is
    # then not guessed; a name no integration reads is a usage error.
    def test_from(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "calls.unknown")
            with mock.patch.object(integrations, "named", return_value=READER) as named:
                self.assertEqual(run("-f", "r", str(source))[0], 0)
            named.assert_called_once_with("r")
        self.assertEqual(run("--from=none", "x")[0], 2)


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
