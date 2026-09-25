"""Run every conformance case through the public interface."""

import json
import logging
import re
import unittest
from pathlib import Path

import mtsv

import living_memory

CONFORMANCE = Path(__file__).resolve().parents[2] / "conformance"

# These match a row of the table of expected reports and rejections,
# and a rejected value's place (conformance/README.md).
_EXPECTED = re.compile(r"^\| `([^`]+)` +\| (not carried|rejected): (.*?) +\|$")
_REJECTED = re.compile(r"value (\d+), pointer `(.*)`")


def expected_table() -> dict[str, tuple[str, str]]:
    # The table gives each case it lists with what is expected and
    # where.
    readme = (CONFORMANCE / "README.md").read_text("utf-8")
    return {
        match[1]: (match[2], match[3])
        for match in map(_EXPECTED.match, readme.splitlines())
        if match
    }


def convert(case: Path) -> dict[str, str]:
    # A case's input values are the lines of name.jsonl, each ended by
    # LF, the last terminator optional; its schema is name.schema.json,
    # and its file pointer name.pointer.json where it names one (JSON
    # Lines, 3. Line Terminator is '\n'; conformance/README.md).
    lines = case.read_bytes().split(b"\n")
    values = lines[:-1] if lines[-1] == b"" else lines
    schema = case.with_suffix(".schema.json").read_bytes()
    pointer_file = case.with_suffix(".pointer.json")
    pointer = pointer_file.read_bytes() if pointer_file.exists() else b""
    return living_memory.convert(values, schema, pointer)


def expected_file(case: Path, name: str) -> Path:
    # The expected file is name.mtsv, the caller naming the one file, or
    # name.VALUE.mtsv for each value of the file member.
    return case.with_name(f"{case.stem}.{name or 'mtsv'}")


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


class TestConformance(unittest.TestCase):
    def assertWritten(self, case: Path) -> list[str]:
        logger = logging.getLogger("living_memory")
        reports = Reports()
        logger.addHandler(reports)
        propagate, logger.propagate = logger.propagate, False
        try:
            files = convert(case)
        finally:
            logger.removeHandler(reports)
            logger.propagate = propagate
        written = {expected_file(case, name) for name in files}
        held = set(case.parent.glob(f"{case.stem}.mtsv"))
        held |= set(case.parent.glob(f"{case.stem}.*.mtsv"))
        self.assertEqual(written, held)
        for name, text in files.items():
            want = expected_file(case, name).read_text("utf-8")
            self.assertEqual(mtsv.loads(text), mtsv.loads(want))
        return reports.messages

    def test_conforming(self):
        for case in cases("conforming"):
            with self.subTest(case=case.name):
                self.assertEqual(self.assertWritten(case), [])

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


if __name__ == "__main__":
    unittest.main()
