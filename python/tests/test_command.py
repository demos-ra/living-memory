"""Tests of _command: how a person runs a conversion."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mtsv
from living_memory import _command
from living_memory.integrations import otlp_json

LINE = (
    '{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"t","spanId":"s",'
    '"name":"n"}]}]}]}\n'
)


def run(argv, stdin=b""):
    """Run the command; return its exit code, output and error."""
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    stderr = io.StringIO()
    code = 0
    with mock.patch("sys.stdin", io.TextIOWrapper(io.BytesIO(stdin))):
        with mock.patch("sys.stdout", stdout), contextlib.redirect_stderr(stderr):
            try:
                _command.run(argv)
            except SystemExit as exit:
                code = exit.code
    stdout.flush()
    return code, stdout.buffer.getvalue(), stderr.getvalue()


class TestRun(unittest.TestCase):
    """run converts a file, or a stream, to MTSV."""

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
        self.assertTrue(code.startswith(f"living-memory:{self.input}: line 1: "))

    def test_missing_file(self):
        missing = Path(self.folder.name) / "missing.jsonl"
        code = run([str(missing)])[0]
        self.assertEqual(code, f"living-memory: {missing}: no such file or directory")

    def test_version(self):
        code, stdout, _ = run(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(stdout.decode("utf-8").startswith("living-memory "))


if __name__ == "__main__":
    unittest.main()
