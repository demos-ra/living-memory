"""Tests of _command: how a person runs a conversion."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mtsv
from living_memory import _command
from living_memory import _otlp_json as otlp_json

LINE = (
    '{"resourceSpans":[{"scopeSpans":[{"spans":[{'
    '"traceId":"5b8efff798038103d269b633813fc60c","spanId":"eee19b7ec3c1b174",'
    '"name":"n"}]}]}]}\n'
)


class Terminal(io.TextIOWrapper):
    """Standard input that is a terminal."""

    def isatty(self):
        return True


def run(argv, stdin=b"", tty=False):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    stderr = io.StringIO()
    code = 0
    wrapper = Terminal if tty else io.TextIOWrapper
    with mock.patch("sys.stdin", wrapper(io.BytesIO(stdin))):
        with mock.patch("sys.stdout", stdout), contextlib.redirect_stderr(stderr):
            try:
                _command.run(argv)
            except SystemExit as exit:
                code = exit.code
    stdout.flush()
    return code, stdout.buffer.getvalue(), stderr.getvalue()


class TestRun(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.input = Path(self.folder.name) / "trace.jsonl"
        self.input.write_text(LINE, encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def test_output_derived(self):
        self.assertEqual(run([str(self.input)])[0], 0)
        with self.input.with_suffix(".mtsv").open("rb") as file:
            self.assertEqual(mtsv.load(file), otlp_json.loads(LINE))

    def test_output_option(self):
        target = Path(self.folder.name) / "out.mtsv"
        self.assertEqual(run(["-o", str(target), str(self.input)])[0], 0)
        self.assertTrue(target.exists())

    def test_standard_streams(self):
        code, stdout, _ = run(["-", "-"], LINE.encode("utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(mtsv.loads(stdout.decode("utf-8")), otlp_json.loads(LINE))

    def test_output_given_twice(self):
        code, _, stderr = run(["-o", "a.mtsv", str(self.input), "b.mtsv"])
        self.assertEqual(code, 2)
        self.assertIn("living-memory: give the output file once", stderr)

    def test_standard_input_needs_output(self):
        self.assertEqual(run(["-"])[0], 2)

    def test_unknown_format(self):
        code, _, stderr = run([str(self.input.with_suffix(".x"))])
        self.assertEqual(code, 2)
        self.assertIn("living-memory: no format for '.x'", stderr)

    def test_line_not_json(self):
        self.input.write_text("x\n", encoding="utf-8")
        code = run([str(self.input)])[0]
        self.assertTrue(code.startswith(f"living-memory:{self.input}:1: "))

    def test_line_not_json_from_standard_input(self):
        code = run(["-", "-"], b"x\n")[0]
        self.assertTrue(code.startswith("living-memory: "))
        self.assertTrue(code.endswith(": line 1"))

    def test_left_behind_is_reported(self):
        self.input.write_text(LINE.replace('"name"', '"extra":1,"name"'))
        code, _, stderr = run([str(self.input)])
        self.assertEqual(code, 0)
        self.assertIn("living-memory: left behind: extra", stderr)

    def test_missing_file(self):
        missing = Path(self.folder.name) / "missing.jsonl"
        code = run([str(missing)])[0]
        self.assertEqual(code, f"living-memory: {missing}: no such file or directory")

    def test_provider_directory(self):
        directory = Path(self.folder.name) / "bodies"
        directory.mkdir()
        (directory / "index.jsonl").write_text("", encoding="utf-8")
        self.assertEqual(run([str(directory)])[0], 0)
        with directory.with_suffix(".mtsv").open("rb") as file:
            self.assertEqual(mtsv.load(file), otlp_json.loads(""))

    def test_directory_output_takes_its_whole_name(self):
        directory = Path(self.folder.name) / "v1.2"
        directory.mkdir()
        (directory / "index.jsonl").write_text("", encoding="utf-8")
        self.assertEqual(run([str(directory)])[0], 0)
        self.assertTrue((Path(self.folder.name) / "v1.2.mtsv").exists())

    def test_directory_of_no_provider(self):
        code, _, stderr = run([self.folder.name])
        self.assertEqual(code, 2)
        self.assertIn("living-memory: no provider for", stderr)

    def test_output_replaced_whole(self):
        # POSIX.1-2017 XSH rename: nothing is left beside the output.
        self.assertEqual(run([str(self.input)])[0], 0)
        self.assertEqual(
            sorted(p.name for p in Path(self.folder.name).iterdir()),
            ["trace.jsonl", "trace.mtsv"],
        )

    def test_version(self):
        code, stdout, _ = run(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(stdout.decode("utf-8").startswith("living-memory "))


class TestInstall(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch("pathlib.Path.home", return_value=Path(self.home.name)),
            mock.patch("sys.platform", "linux"),
            mock.patch.dict("os.environ", {"XDG_DATA_HOME": ""}),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.home.cleanup()

    def test_no_such_plugin(self):
        code, _, stderr = run(["--install=x"])
        self.assertEqual(code, 2)
        self.assertIn("living-memory: no plugin 'x'", stderr)

    def test_no_input_with_install(self):
        self.assertEqual(run(["--install=claude-code", "a.jsonl"])[0], 2)

    def test_no_terminal_no_change(self):
        code = run(["--install=claude-code"])[0]
        self.assertEqual(
            code, "living-memory: --install asks first, and needs a terminal"
        )
        self.assertEqual(list(Path(self.home.name).iterdir()), [])

    def test_no_by_default(self):
        with mock.patch("builtins.input", return_value=""):
            with mock.patch("subprocess.run") as commands:
                self.assertEqual(run(["--install=claude-code"], tty=True)[0], 0)
        commands.assert_not_called()
        self.assertEqual(list(Path(self.home.name).iterdir()), [])

    def test_yes(self):
        home = Path(self.home.name)
        data = home / ".local" / "share" / "living-memory"
        with mock.patch("builtins.input", return_value="y"):
            with mock.patch("subprocess.run") as commands:
                self.assertEqual(run(["--install=claude-code"], tty=True)[0], 0)
        self.assertEqual(commands.call_count, 2)
        bodies = data / "anthropic" / "claude_code" / "raw_api_bodies"
        self.assertEqual((bodies / "index.jsonl").read_bytes(), b"")
        self.assertEqual(oct(data.stat().st_mode & 0o777), "0o700")
        settings = (home / ".claude" / "settings.json").read_text()
        self.assertIn(f"file:{bodies}", settings)


if __name__ == "__main__":
    unittest.main()
