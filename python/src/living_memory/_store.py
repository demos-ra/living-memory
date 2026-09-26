"""The output kept as its sheets' files, each only appended to."""

__all__ = ["append", "locked", "repair"]

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mtsv

# The file a conversion holds its lock on, beside the sheets' files.
_LOCK = ".lock"
# The key column that names each record's value by its position in the
# input, its pointer's first token.
_POINTER = "pointer"


@contextmanager
def locked(folder: Path) -> Iterator[None]:
    # One conversion at a time: each waits for an exclusive lock on the
    # output, which keeps every other conversion from locking it.
    folder.mkdir(exist_ok=True)
    with open(folder / _LOCK, "a") as fp:
        fcntl.lockf(fp, fcntl.LOCK_EX)
        yield


def repair(folder: Path, names: list[str]) -> list[dict[str, str]]:
    # Before a conversion appends, a line not ended, and a record whose
    # value's position the sheet of the input values does not hold, are
    # left out; that sheet's records are returned, each by its header's
    # names, their number the position of the next value.
    paths = _paths(folder, names)
    header, records = _kept(paths[0], None)
    for path in paths[1:]:
        _kept(path, len(records))
    return [dict(zip(header, record)) for record in records]


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
            written = written.split("\n", 2)[2]
        with open(path, "ab") as fp:
            fp.write(written.encode("utf-8"))


def _paths(folder: Path, names: list[str]) -> list[Path]:
    # A sheet's file is named by its zero-based place in the file's
    # order and its sheet name.
    width = len(str(len(names) - 1))
    return [
        folder / f"{place:0{width}} {name}.mtsv" for place, name in enumerate(names)
    ]


def _kept(path: Path, held: int | None) -> tuple[list[str], list[list[str]]]:
    # A sheet's header and the records it keeps: the whole lines, and of
    # those, where held is given, the records whose value's position is
    # held; the file is rewritten, beside its name and renamed onto it,
    # only where something is left out (POSIX.1-2017, rename).
    if not path.exists():
        return [], []
    text = path.read_bytes().decode("utf-8")
    ended = text[: text.rfind("\n") + 1]
    lines = ended.split("\n")[:-1]
    header = lines[1].split("\t") if len(lines) > 1 else []
    records = [line.split("\t") for line in lines[2:]]
    if held is not None and _POINTER in header:
        column = header.index(_POINTER)
        records = [r for r in records if int(r[column].split("/")[1]) < held]
    kept = "\n".join([*lines[:2], *("\t".join(r) for r in records)]) + "\n"
    if len(lines) < 2:
        kept = ""
    if kept != text:
        _replace(path, kept.encode("utf-8"))
    return header, records


def _replace(path: Path, data: bytes) -> None:
    handle, written = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with open(handle, "wb") as fp:
        fp.write(data)
    os.replace(written, path)
