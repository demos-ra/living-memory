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
    # appended sheet by sheet in the file's order (value.1).
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


if __name__ == "__main__":
    unittest.main()
