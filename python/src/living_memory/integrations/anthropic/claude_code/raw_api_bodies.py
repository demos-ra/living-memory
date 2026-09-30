"""Claude Code's recording of the raw API bodies, as input values."""

__all__ = [
    "FILE",
    "input_of",
    "inputs",
    "lines",
    "read",
    "schema",
    "spent",
    "values",
]

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from living_memory import _json, _json_pointer, _json_schema
from living_memory._json import Number
from living_memory.integrations.anthropic import messages

# In file mode each successful response appends one line to
# <dir>/index.jsonl (raw_api_bodies.mtsv › recording.1).
FILE = "index.jsonl"
# The sheet of the input values is named after the recording.
_TITLE = "raw_api_bodies"
# The request's members that hold a list of units, each compared with
# the parent's at its pointer, and the response's content, the message
# that follows (raw_api_bodies.mtsv › values.1, values.3).
_LISTS = ("system", "tools")
_RESPONSE = "content"
# A value's names for its index line and for the index line of the
# request it extends, names the living-memory specification's Fields do
# not give (raw_api_bodies.mtsv › values.2, values.4).
_LINE = "index_line"
_EXTENDS = "extends"
# The index line's members that name the request (values.4).
_INDEX = ("session_id", "query_source", "model", "timestamp")
# Index lines count from 1, so 0 names no parent (JSON Lines;
# raw_api_bodies.mtsv › values.2).
_NO_PARENT = 0
# An input is named by its date, the first ten characters of an ISO 8601
# timestamp, then '/', then its session_id (values.7).
_DATE = 10
_SEPARATOR = "/"
# The files an index line names (recording.1).
_FILES = ("request_file", "response_file")


@dataclass(frozen=True)
class _Request:
    # A request as it is compared: its index line and thread, its
    # messages as they are kept, then its response as the message of its
    # role and content that follows, and its system blocks and tools,
    # each as it is kept.
    line: int
    thread: tuple[str, str]
    sequence: list[Any]
    lists: dict[str, list[Any]]


def schema() -> bytes:
    """Return the schema of the input values, as a JSON text.

    A value names its request by index_line, session_id, query_source,
    model and timestamp; names as extends the request it extends and as
    kept how many of its messages it keeps; counts its system blocks and
    tools; and holds the units the request it extends does not, each at
    its pointer in the request, and the response's members kept beside
    its content (raw_api_bodies.mtsv › values.2-4).
    """
    count = {"type": "integer", "minimum": Number("0")}
    pointer = {
        "type": "object",
        "properties": {"pointer": {"type": "string"}},
        "required": ["pointer"],
        "additionalProperties": False,
    }
    unit = {
        "type": "object",
        "properties": {"request": pointer, **messages.units()},
        "required": ["request"],
        "additionalProperties": False,
    }
    counts = {
        "type": "object",
        "properties": {name: count for name in _LISTS},
        "required": list(_LISTS),
        "additionalProperties": False,
    }
    root = {
        "title": _TITLE,
        "type": "object",
        "properties": {
            _LINE: {"type": "integer", "minimum": Number("1")},
            **{name: {"type": "string"} for name in _INDEX},
            _EXTENDS: count,
            "kept": count,
            "count": counts,
            "units": {"type": "array", "items": unit},
            **messages.ending(),
        },
        "required": [_LINE, *_INDEX, _EXTENDS, "kept", "count", "units"],
        "additionalProperties": False,
        "definitions": messages.definitions(),
    }
    return _json.encode(root)


def lines(path: Path, start: int) -> list[tuple[int, dict[str, Any]]]:
    """Return a recording's index lines after the first start of them.

    Each is its number, counted from 1, and its JSON value; only these
    lines are decoded. A folder without index.jsonl holds none
    (raw_api_bodies.mtsv › recording.1, recording.8, values.6).
    """
    file = path / FILE
    if not file.is_file():
        return []
    found = _split(file.read_bytes())
    return [
        (number, _json.decode(found[number - 1]))
        for number in range(start + 1, len(found) + 1)
    ]


def inputs(index: list[tuple[int, dict[str, Any]]], names: list[str]) -> list[str]:
    """Return the names of the inputs that index lines give values to.

    index -- index lines, as lines gives them
    names -- the names of the inputs the output holds

    One for each session with a line that names a request, in the order
    of those lines: its name among names, else the date of that line's
    timestamp, then '/', then its session_id (raw_api_bodies.mtsv ›
    values.1, values.7).
    """
    found: dict[str, str] = {}
    for _, entry in index:
        session = entry["session_id"]
        if "request_file" in entry and session not in found:
            held = input_of(names, session)
            date = entry["timestamp"][:_DATE]
            found[session] = held or f"{date}{_SEPARATOR}{session}"
    return list(found.values())


def read(
    index: list[tuple[int, dict[str, Any]]], held: list[dict[str, str]], start: int
) -> int:
    """Return how many index lines are read.

    index -- the index lines after the first start, as lines gives them
    held -- the records the outputs of their inputs hold, each by its
        header's names

    Those before the first line that names a request its input's
    output does not hold (raw_api_bodies.mtsv › values.6).
    """
    numbers = {int(record[_LINE]) for record in held}
    for number, entry in index:
        if "request_file" in entry and number not in numbers:
            return number - 1
    return index[-1][0] if index else start


def input_of(names: list[str], session: str) -> str | None:
    """Return the name of a session's input among names, or None."""
    return next((n for n in names if n.split(_SEPARATOR)[-1] == session), None)


def values(
    path: Path,
    name: str,
    held: list[dict[str, str]] | None,
    index: list[tuple[int, dict[str, Any]]],
) -> Iterator[bytes]:
    """Yield the input values of one of a recording's inputs.

    path -- the recording's folder, which holds index.jsonl
    name -- the input's name, as inputs gives it
    held -- the records of the input's output's sheet of the input
        values, each by its header's names, or None where it holds none
    index -- the index lines read, as lines gives them

    One value for each of the session's requests among the lines read,
    in the index's order, after the last one held, until a line whose
    files are not yet written; with none, no file is read. Raise
    ValueError where a file of the latest request of a thread held is
    absent (raw_api_bodies.mtsv › recording.4, recording.5, recording.7,
    values.1-7).
    """
    after = _after(held)
    new = [
        (number, entry)
        for number, entry in _session(index, name)
        if number > after and "request_file" in entry
    ]
    if not new:
        return
    tips = _tips(path, held or [], index)
    for number, entry in new:
        if not _written(entry, path):
            return
        request, response = _bodies(entry, path)
        current = _request(number, entry, request, response)
        parent = _parent(current, tips)
        yield _value(entry, (request, response), current, parent)
        tips[current.thread] = current


def spent(
    path: Path,
    name: str,
    held: list[dict[str, str]],
    index: list[tuple[int, dict[str, Any]]],
) -> list[Path]:
    """Return the request and response files an input's output spent.

    The files of the session's lines read that the output holds or that
    name no request, and of the latest request of each thread the output
    held before them, but those of the latest request of each thread it
    now holds, which the next conversion reads again; index.jsonl is
    never among them (raw_api_bodies.mtsv › recording.5, recording.6,
    values.6).
    """
    session = _session(index, name)
    after = _after(held)
    read_now = {number for number, _ in session}
    before = _latest([r for r in held if int(r[_LINE]) not in read_now])
    entries = dict(session)
    entries.update(_entries(path, [n for n in before.values() if n not in entries]))
    done = {
        number
        for number, entry in session
        if number <= after or "request_file" not in entry
    }
    kept = set(_latest(held).values())
    return [
        _file(entries[number], field, path)
        for number in sorted((done | set(before.values())) - kept)
        for field in _FILES
        if field in entries[number]
    ]


def _session(
    index: list[tuple[int, dict[str, Any]]], name: str
) -> list[tuple[int, dict[str, Any]]]:
    # The index lines of one input's session (values.7).
    session = name.split(_SEPARATOR)[-1]
    return [(n, entry) for n, entry in index if entry["session_id"] == session]


def _after(held: list[dict[str, str]] | None) -> int:
    # The last index line the output holds, 0 where it holds none
    # (values.6).
    return max((int(record[_LINE]) for record in held or []), default=0)


def _split(data: bytes) -> list[bytes]:
    # The index's lines, not yet decoded, the first line 1; a terminator
    # after the last one is optional (JSON Lines).
    found = data.split(b"\n")
    if found and found[-1] == b"":
        found.pop()
    return found


def _entries(path: Path, numbers: list[int]) -> dict[int, dict[str, Any]]:
    # The index lines of the numbers given, each decoded alone
    # (values.6).
    if not numbers:
        return {}
    found = _split((path / FILE).read_bytes())
    for number in numbers:
        if number > len(found):
            raise ValueError(f"{path / FILE}: holds no line {number}")
    return {number: _json.decode(found[number - 1]) for number in numbers}


def _latest(held: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    # The index line of the latest request of each thread held.
    found: dict[tuple[str, str], int] = {}
    for record in held:
        thread = (record["session_id"], record["query_source"])
        found[thread] = max(found.get(thread, 0), int(record[_LINE]))
    return found


def _file(entry: dict[str, Any], field: str, path: Path) -> Path:
    # A path that is not absolute is in the recording's folder
    # (raw_api_bodies.mtsv › recording.3).
    named = Path(entry[field])
    return named if named.is_absolute() else path / named


def _written(entry: dict[str, Any], path: Path) -> bool:
    # Whether every file a line names is written (recording.4).
    return all(_file(entry, f, path).is_file() for f in _FILES if f in entry)


def _bodies(entry: dict[str, Any], path: Path) -> tuple[Any, Any]:
    # A request line's request and response, each read from its file.
    request, response = (
        _json.decode(_file(entry, f, path).read_bytes()) for f in _FILES
    )
    return request, response


def _request(
    number: int, entry: dict[str, Any], request: Any, response: Any
) -> _Request:
    # A request's messages, each whole as messages.mtsv keeps it, then
    # its response as the message that follows, of its role, assistant,
    # and its content, which is all it holds; and its system blocks and
    # tools, each as it is kept (values.3).
    said = [messages.kept("messages", m) for m in request.get("messages", [])]
    answer = {
        "role": "assistant",
        "content": messages.kept(_RESPONSE, response[_RESPONSE]),
    }
    lists = {
        name: [messages.kept(name, unit) for unit in request.get(name) or []]
        for name in _LISTS
        if isinstance(request.get(name), list)
    }
    thread = (entry["session_id"], entry["query_source"])
    return _Request(number, thread, [*said, answer], lists)


def _tips(
    path: Path, held: list[dict[str, str]], index: list[tuple[int, dict[str, Any]]]
) -> dict[tuple[str, str], _Request]:
    # The latest request of each thread held, its line taken from the
    # lines read or decoded alone, read again from its files; a file of
    # one that is absent is rejected, named (values.6, recording.7).
    latest = _latest(held)
    known = dict(index)
    entries = {n: known[n] for n in latest.values() if n in known}
    entries.update(_entries(path, [n for n in latest.values() if n not in known]))
    found = {}
    for thread, number in latest.items():
        entry = entries[number]
        for field in _FILES:
            if not _file(entry, field, path).is_file():
                raise ValueError(
                    f"{_file(entry, field, path)}: absent, a file of the latest"
                    " request of a thread held"
                )
        request, response = _bodies(entry, path)
        found[thread] = _request(number, entry, request, response)
    return found


def _parent(current: _Request, tips: dict[tuple[str, str], _Request]) -> tuple:
    # Of the latest request of each thread of the session, the one whose
    # messages, then response, the request's messages agree with
    # furthest, the later where two agree as far; none where none agrees
    # with its first message (values.2).
    own = current.sequence[:-1]
    best: tuple[int, int, _Request | None] = (0, _NO_PARENT, None)
    for tip in tips.values():
        if tip.thread[0] != current.thread[0]:
            continue
        agree = 0
        for mine, theirs in zip(own, tip.sequence):
            if not _json_schema.equal(mine, theirs):
                break
            agree += 1
        best = max(best, (agree, tip.line, tip), key=lambda b: (b[0], b[1]))
    return best if best[0] else (0, _NO_PARENT, None)


def _value(
    entry: dict[str, Any],
    bodies: tuple[Any, Any],
    current: _Request,
    parent: tuple,
) -> bytes:
    # One input value: the request's names, its parent and how many of
    # its messages it keeps, its counts, the units the parent does not
    # hold, and the response's members kept beside its content
    # (values.2-4).
    request, response = bodies
    kept, parent_line, tip = parent
    units: list[dict[str, Any]] = []
    system = request.get("system")
    if isinstance(system, str):
        units.append(_unit("/system", "system", system))
    for name in _LISTS:
        before = tip.lists.get(name, []) if tip else []
        for position, unit in enumerate(current.lists.get(name, [])):
            if position >= len(before) or not _json_schema.equal(
                unit, before[position]
            ):
                units.append(
                    _unit(_json_pointer.pointer(f"/{name}", position), name, unit)
                )
    said = request.get("messages", [])
    for position in range(kept, len(said)):
        message = messages.kept("messages", said[position])
        units.append(
            _unit(_json_pointer.pointer("/messages", position), "messages", message)
        )
    answer = messages.kept(_RESPONSE, response[_RESPONSE])
    units.append(
        _unit(_json_pointer.pointer("/messages", len(said)), _RESPONSE, answer)
    )
    value: dict[str, Any] = {
        _LINE: Number(str(current.line)),
        **{name: entry[name] for name in _INDEX},
        _EXTENDS: Number(str(parent_line)),
        "kept": Number(str(kept)),
        "count": {
            name: Number(str(len(current.lists.get(name, [])))) for name in _LISTS
        },
        "units": units,
    }
    for name in messages.ending():
        if response.get(name) not in (None, [], {}):
            value[name] = response[name]
    return _json.encode(value)


def _unit(pointer: str, member: str, unit: Any) -> dict[str, Any]:
    # A unit at its JSON Pointer in the request, under the member that
    # holds it (values.3).
    return {"request": {"pointer": pointer}, member: unit}
