"""The output kept as its sheets' files, each only appended to."""

__all__ = ["append", "held", "locked", "repair"]

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mtsv

from living_memory import _rename

# The file a conversion holds its lock on, beside the sheets' files.
_LOCK = ".lock"
# The key column that names each record's value by its position in the
# input, its pointer's first token.
_POINTER = "pointer"
# A sheet's file begins with two lines before its records: the FF line
# with the sheet name, then the header.
_HEAD = 2


@contextmanager
def locked(folder: Path) -> Iterator[None]:
    # One conversion at a time: each waits for an exclusive lock on the
    # output, which keeps every other conversion from locking it.
    folder.mkdir(exist_ok=True)
    with open(folder / _LOCK, "a") as fp:
        fcntl.lockf(fp, fcntl.LOCK_EX)
        yield


def repair(folder: Path, names: list[str]) -> None:
    # Before a conversion appends, a line not ended, and a record whose
    # value's position the sheet of the input values does not hold, are
    # left out.
    paths = _paths(folder, names)
    for path in paths:
        _end(path)
    count = len(_lines(paths[0])[_HEAD:])
    for path in paths[1:]:
        _within(path, count)


def held(folder: Path, names: list[str]) -> list[dict[str, str]]:
    # The records of the sheet of the input values, each by its header's
    # names; their number is the position of the next value.
    lines = _lines(_paths(folder, names)[0])
    if len(lines) < _HEAD:
        return []
    header = lines[_HEAD - 1].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[_HEAD:]]


def append(folder: Path, text: str, names: list[str]) -> None:
    # Each sheet's new records are written at the end of its file, the
    # file opened for appending, with its header when it is created;
    # the sheet of the input values last, so a conversion cut short is
    # found by it.
    paths = _paths(folder, names)
    sheets = mtsv.loads(text)
    for sheet in sorted(sheets, key=lambda s: names.index(s["sheet name"]) == 0):
        path = paths[names.index(sheet["sheet name"])]
        written = mtsv.dumps([sheet])
        if path.exists() and path.stat().st_size:
            written = written.split("\n", _HEAD)[_HEAD]
        with open(path, "ab") as fp:
            fp.write(written.encode("utf-8"))


def _paths(folder: Path, names: list[str]) -> list[Path]:
    # A sheet's file is named by its zero-based place in the file's
    # order and its sheet name.
    width = len(str(len(names) - 1))
    return [
        folder / f"{place:0{width}} {name}.mtsv" for place, name in enumerate(names)
    ]


def _lines(path: Path) -> list[str]:
    # A sheet file's whole lines, each ended by LF.
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8")
    return text[: text.rfind("\n") + 1].split("\n")[:-1]


def _end(path: Path) -> None:
    # A line not ended is left out; a file left with less than its FF
    # line and header is left empty.
    if not path.exists():
        return
    lines = _lines(path)
    kept = "".join(f"{line}\n" for line in lines) if len(lines) >= _HEAD else ""
    if kept != path.read_bytes().decode("utf-8"):
        _rename.replace(path, kept.encode("utf-8"))


def _within(path: Path, count: int) -> None:
    # A record whose value's position the sheet of the input values
    # does not hold is left out.
    lines = _lines(path)
    if len(lines) < _HEAD:
        return
    column = lines[_HEAD - 1].split("\t").index(_POINTER)
    records = lines[_HEAD:]
    kept = [r for r in records if int(r.split("\t")[column].split("/")[1]) < count]
    if kept != records:
        text = "".join(f"{line}\n" for line in [*lines[:_HEAD], *kept])
        _rename.replace(path, text.encode("utf-8"))
