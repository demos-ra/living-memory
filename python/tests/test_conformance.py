"""Run every conformance case: a converter's through the public
interface, a data bank's through the command."""

import contextlib
import io
import json
import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mtsv

import living_memory
from living_memory import _command, integrations

CONFORMANCE = Path(__file__).resolve().parents[2] / "conformance"

# These match a row of the table of expected reports and rejections,
# and a rejected value's place (conformance/README.md).
_EXPECTED = re.compile(r"^\| `([^`]+)` +\| (not carried|rejected): (.*?) +\|$")
_REJECTED = re.compile(r"value (\d+), pointer `(.*?)`")


def expected_table() -> dict[str, tuple[str, str]]:
    # The table gives each case it lists with what is expected and
    # where.
    readme = (CONFORMANCE / "README.md").read_text("utf-8")
    return {
        match[1]: (match[2], match[3])
        for match in map(_EXPECTED.match, readme.splitlines())
        if match
    }


def convert(case: Path) -> str:
    # A case's input values are the lines of name.jsonl, each ended by
    # LF, the last terminator optional; its schema is name.schema.json
    # (JSON Lines, 3. Line Terminator is '\n'; conformance/README.md).
    lines = case.read_bytes().split(b"\n")
    values = lines[:-1] if lines[-1] == b"" else lines
    schema = case.with_suffix(".schema.json").read_bytes()
    return living_memory.convert(values, schema)


class Reports(logging.Handler):
    # The handler keeps what the converter reports as not carried, in
    # order.
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def cases(folder: str) -> list[Path]:
    return sorted((CONFORMANCE / folder).glob("*.jsonl"))


def in_two_parts(case: Path) -> list[dict]:
    # The values before the middle, then the rest, each part converted
    # with its values' positions in the whole input, and their sheets
    # appended sheet by sheet in the file's order (value.2).
    lines = case.read_bytes().split(b"\n")
    values = lines[:-1] if lines[-1] == b"" else lines
    schema = case.with_suffix(".schema.json").read_bytes()
    middle = len(values) // 2
    parts = [
        mtsv.loads(living_memory.convert(values[:middle], schema)),
        mtsv.loads(living_memory.convert(values[middle:], schema, middle)),
    ]
    joined = []
    for name in living_memory.sheets(schema):
        found = [s for part in parts for s in part if s["sheet name"] == name]
        if found:
            records = [record for s in found for record in s["records"]]
            joined.append({**found[0], "records": records})
    return joined


class TestConformance(unittest.TestCase):
    def assertWritten(self, case: Path) -> list[str]:
        logger = logging.getLogger("living_memory")
        reports = Reports()
        logger.addHandler(reports)
        propagate, logger.propagate = logger.propagate, False
        try:
            text = convert(case)
        finally:
            logger.removeHandler(reports)
            logger.propagate = propagate
        want = case.with_suffix(".mtsv").read_text("utf-8")
        self.assertEqual(mtsv.loads(text), mtsv.loads(want))
        return reports.messages

    def test_conforming(self):
        for case in cases("conforming"):
            with self.subTest(case=case.name):
                self.assertEqual(self.assertWritten(case), [])

    def test_parts(self):
        for case in cases("conforming"):
            if case.read_bytes().rstrip(b"\n").count(b"\n") == 0:
                continue
            with self.subTest(case=case.name):
                want = case.with_suffix(".mtsv").read_text("utf-8")
                self.assertEqual(in_two_parts(case), mtsv.loads(want))

    def test_cannot_be_represented(self):
        table = expected_table()
        for case in cases("cannot-be-represented"):
            with self.subTest(case=case.name):
                kind, pointers = table[case.stem]
                self.assertEqual(kind, "not carried")
                want = [f"not carried: {p}" for p in re.findall(r"`(.*?)`", pointers)]
                self.assertEqual(self.assertWritten(case), want)

    def test_non_conforming(self):
        table = expected_table()
        for case in cases("non-conforming"):
            with self.subTest(case=case.name):
                kind, place = table[case.stem]
                self.assertEqual(kind, "rejected")
                with self.assertRaises(living_memory.NonConformingError) as raised:
                    convert(case)
                error = raised.exception
                match = _REJECTED.fullmatch(place)
                if match is None:
                    self.assertEqual(place, "the module specification")
                    self.assertNotIsInstance(
                        error, living_memory.NonConformingInputError
                    )
                else:
                    self.assertIsInstance(error, living_memory.NonConformingInputError)
                    self.assertEqual(error.position, int(match[1]))
                    self.assertEqual(error.pointer, json.loads(match[2]))


class Part:
    # A reader of one conformance input, stored in parts: it gives the
    # values after those its data bank stores, to the end of the part
    # (conformance/README.md, A data bank's steps).
    def __init__(self, case: Path, schema: Path) -> None:
        lines = case.read_bytes().split(b"\n")
        self.all = lines[:-1] if lines[-1] == b"" else lines
        self.end = len(self.all) // 2
        self.schema_text = schema.read_bytes()

    def schema(self) -> bytes:
        return self.schema_text

    def values(self, path: Path, held: list | None) -> list[bytes]:
        stored = sum(1 for r in held or [] if r["pointer"].count("/") == 1)
        return self.all[stored : self.end]


def bank(reader: Part, *argv: str) -> tuple[int, str, str]:
    # The data bank's command, its reader the case's, with what it
    # writes to standard output and standard error.
    data = io.BytesIO()
    out, err = io.TextIOWrapper(data, "utf-8"), io.StringIO()
    with mock.patch.object(integrations, "reader", return_value=reader):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                _command.main(list(argv))
                code = 0
            except SystemExit as exit:
                code = exit.code if isinstance(exit.code, int) else 1
                err.write(exit.code if isinstance(exit.code, str) else "")
    out.flush()
    return code, data.getvalue().decode("utf-8"), err.getvalue()


# The request of step 8, where a case has one (conformance/README.md,
# communication.4).
REQUESTS = {"storage.3.parts": ["--values=1-2", "--places=1;3"]}


class TestDataBank(unittest.TestCase):
    def expected(self, name: str) -> str:
        return (CONFORMANCE / "communicated" / name).read_text("utf-8")

    # conformance.2, storage.1-6, communication.1-6: each case stored in
    # two parts and communicated in the README's eight steps.
    def test_communicated(self):
        table = expected_table()
        for case in cases("communicated"):
            if case.stem.count(".") > 2:
                continue
            with self.subTest(case=case.name), tempfile.TemporaryDirectory() as tmp:
                name = case.stem
                source = str(shutil.copy(case, Path(tmp) / case.name))
                reader = Part(case, case.with_suffix(".schema.json"))
                steps = {}
                self.assertEqual(bank(reader, source)[0], 0)
                steps["names"] = bank(reader, "--names", source)[1]
                steps["new-1"] = bank(reader, f"--new={name}", source)[1]
                reader.end = len(reader.all)
                code, _, err = bank(reader, source)
                if name in table:
                    self.assertEqual(code, 1)
                    self.assertIn(_rejected(table[name][1]), err)
                steps["new-2"] = bank(reader, f"--new={name}", source)[1]
                steps["again"] = bank(reader, f"--new={name}", source)[1]
                steps["all"] = bank(reader, f"--filter={name}", source)[1]
                if name in REQUESTS:
                    argv = [f"--filter={name}", *REQUESTS[name], source]
                    steps["request"] = bank(reader, *argv)[1]
                for step, text in steps.items():
                    with self.subTest(step=step):
                        self.assertEqual(text, self.expected(f"{name}.{step}.mtsv"))

    # storage.2: two inputs stored apart in one data bank, each whole.
    def test_two_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = f"{tmp}/bank.mtsv"
            for part in ("a", "b"):
                case = (
                    CONFORMANCE / "communicated" / f"storage.2.two-inputs.{part}.jsonl"
                )
                reader = Part(case, case.with_name("storage.2.two-inputs.schema.json"))
                reader.end = len(reader.all)
                source = str(shutil.copy(case, Path(tmp) / case.name))
                self.assertEqual(bank(reader, "-o", output, source)[0], 0)
            names = bank(reader, "-o", output, "--names", source)[1]
            self.assertEqual(names, self.expected("storage.2.two-inputs.names.mtsv"))
            for part in ("a", "b"):
                with self.subTest(part=part):
                    name = f"storage.2.two-inputs.{part}"
                    text = bank(reader, "-o", output, f"--filter={name}", source)[1]
                    self.assertEqual(text, self.expected(f"{name}.all.mtsv"))


def _rejected(expected: str) -> str:
    # A part rejected at a value is reported by the value's position and
    # its pointer, as the converter names it.
    match = re.search(r"value (\d+), pointer `(.*)`", expected)
    return f"value {match[1]}, {match[2]}"


if __name__ == "__main__":
    unittest.main()
