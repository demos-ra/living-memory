"""The data bank's storage: each input's sheets kept as files, appended
to, and cut back to the values stored whole after a conversion cut
short; only the command reads them."""

__all__ = [
    "add_input",
    "append",
    "communicated",
    "inputs",
    "locked",
    "mark_communicated",
    "mark_read",
    "read",
    "repair",
    "stored",
]

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mtsv

from living_memory import _rename, _storage
from living_memory._storage import Stored

# The file a conversion holds its lock on, beside the sheets' files.
_LOCK = ".lock"
# A sheet's file begins with two lines before its records: the FF line
# with the sheet name, then the header.
_HEAD = 2
# The file beside an input's storage that holds how many of its values
# have been communicated (spec › communication.6).
_COMMUNICATED = ".communicated"
# The file beside the storage that holds how many lines of its source
# have been read.
_READ = ".read"
# The file that names the inputs stored, one to a line, in the order
# first stored.
_INPUTS = ".inputs"
# Every sheet's file ends with this extension (MTSV draft, Media Type
# Registration).
_EXTENSION = ".mtsv"


@contextmanager
def locked(folder: Path) -> Iterator[None]:
    # One conversion at a time: each waits for an exclusive lock on the
    # storage, which keeps every other conversion from locking it.
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / _LOCK, "a") as fp:
        fcntl.lockf(fp, fcntl.LOCK_EX)
        yield


def stored(folder: Path, names: list[str]) -> Stored:
    # An input's stored sheets, each with its place among every sheet
    # the schema gives; a sheet with no file holds no record.
    found: Stored = []
    for place, path in enumerate(_paths(folder, names)):
        lines = _lines(path)
        if len(lines) >= _HEAD:
            found.append((place, mtsv.loads("".join(f"{l}\n" for l in lines))[0]))
    return found


def repair(folder: Path, names: list[str]) -> None:
    # Before a part is inserted, a line not ended, and the records of a
    # value not wholly stored, are no part of the storage (spec ›
    # storage.4).
    for path in _paths(folder, names):
        _end(path)
    held = stored(folder, names)
    kept = _storage.whole(held, _storage.count(held))
    for (place, sheet), (_, whole) in zip(held, kept):
        if whole["records"] != sheet["records"]:
            _rename.replace(_paths(folder, names)[place], _text(whole))


def append(folder: Path, text: str, names: list[str]) -> None:
    # Each sheet's new records are written at the end of its file, the
    # file opened for appending, with its FF line and header when it is
    # created; the sheet of the input values last, so a conversion cut
    # short is found by it (spec › storage.3, storage.4).
    paths = _paths(folder, names)
    sheets = mtsv.loads(text)
    if sheets:
        folder.mkdir(parents=True, exist_ok=True)
    for sheet in sorted(sheets, key=lambda s: names.index(s["sheet name"]) == 0):
        path = paths[names.index(sheet["sheet name"])]
        written = _text(sheet).decode("utf-8")
        if path.exists() and path.stat().st_size:
            written = written.split("\n", _HEAD)[_HEAD]
        with open(path, "ab") as fp:
            fp.write(written.encode("utf-8"))


def inputs(folder: Path) -> list[str]:
    # The names of the inputs stored, each the path of the folder of its
    # sheets' files within the storage, in the order first stored (spec
    # › storage.2, communication.3).
    path = folder / _INPUTS
    return path.read_text("utf-8").splitlines() if path.exists() else []


def add_input(folder: Path, name: str) -> None:
    # An input stored for the first time is added after those stored
    # before it, the file replaced whole.
    held = inputs(folder)
    if name not in held:
        text = "".join(f"{each}\n" for each in [*held, name])
        _rename.replace(folder / _INPUTS, text.encode("utf-8"))


def communicated(folder: Path) -> int:
    # How many of an input's values have been communicated.
    return _count(folder / _COMMUNICATED)


def mark_communicated(folder: Path, count: int) -> None:
    _mark(folder, _COMMUNICATED, count)


def read(folder: Path) -> int:
    # How many lines of the storage's source have been read.
    return _count(folder / _READ)


def mark_read(folder: Path, count: int) -> None:
    _mark(folder, _READ, count)


def _count(path: Path) -> int:
    # A count kept in a file, 0 where there is none.
    return int(path.read_text("utf-8")) if path.exists() else 0


def _mark(folder: Path, name: str, count: int) -> None:
    # A count kept beside the storage, the file replaced whole.
    if folder.is_dir():
        _rename.replace(folder / name, f"{count}\n".encode("utf-8"))


def _paths(folder: Path, names: list[str]) -> list[Path]:
    # A sheet's file is named by its zero-based place and its sheet
    # name, the place written with as many digits as the last place, so
    # the files in the order of their names are the file's sheets in
    # order.
    width = len(str(len(names) - 1))
    return [
        folder / f"{place:0{width}} {name}{_EXTENSION}"
        for place, name in enumerate(names)
    ]


def _text(sheet: dict) -> bytes:
    return mtsv.dumps([sheet]).encode("utf-8")


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
