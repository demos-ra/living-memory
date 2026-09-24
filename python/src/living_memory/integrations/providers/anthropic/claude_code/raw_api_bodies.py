"""How Claude Code's raw API bodies of Messages API calls are read."""

__all__ = ["load", "logs_data", "RawAPIBodiesDecodeError"]

import json
import logging
import os
from pathlib import Path
from typing import Any

from living_memory._json import Number, decode
from living_memory.integrations import otlp_json
from living_memory.integrations.providers.anthropic import messages

# raw_api_bodies.mtsv › index.1.
_INDEX = "index.jsonl"
# raw_api_bodies.mtsv › event.1.
_EVENT = "gen_ai.client.inference.operation.details"
# raw_api_bodies.mtsv › index.2.
_NAMESPACE = "anthropic.claude_code."
# raw_api_bodies.mtsv › index.4: the members naming the files.
_FILES = ("request_file", "response_file")

_logger = logging.getLogger("living_memory.integrations")

_Document = dict[str, Any] | None


class RawAPIBodiesDecodeError(ValueError):
    """Subclass of ValueError with the following additional properties:

    msg: The unformatted error message
    lineno: The line of the index, the first being line 1
    """

    def __init__(self, msg: str, lineno: int) -> None:
        super().__init__(f"{msg}: line {lineno}")
        self.msg = msg
        self.lineno = lineno

    def __reduce__(self) -> tuple[type, tuple[str, int]]:
        return self.__class__, (self.msg, self.lineno)


def load(path: str | os.PathLike[str], /) -> list[dict[str, Any]]:
    """Read MTSV sheets from a folder of Claude Code's raw API bodies.

    Raise RawAPIBodiesDecodeError for raw API bodies not conforming.
    """
    # raw_api_bodies.mtsv › conformance.1, index.4: one line for each
    # line of the index, so the core's line is the index's.
    lines = "".join(_encode(data) + "\n" for data in logs_data(path))
    try:
        return otlp_json.loads(lines)
    except otlp_json.OTLPDecodeError as error:
        raise RawAPIBodiesDecodeError(error.msg, error.lineno) from None


def logs_data(path: str | os.PathLike[str], /) -> list[dict[str, Any]]:
    """Return one LogsData for each line of the raw API bodies' index.

    Raise RawAPIBodiesDecodeError for raw API bodies not conforming.
    """
    # raw_api_bodies.mtsv › index.1, index.4.
    directory = Path(path)
    found = []
    for index, line in enumerate(_index_lines(directory)):
        try:
            entry, request, response = _documents(directory, line)
        except ValueError as error:
            raise RawAPIBodiesDecodeError(str(error), index + 1) from None
        for member, document in zip(_FILES, (request, response)):
            if document is None:
                _logger.warning("line %d: %s not held", index + 1, member)
        try:
            found.append(_logs_data(entry, request, response))
        except ValueError as error:
            raise RawAPIBodiesDecodeError(str(error), index + 1) from None
    return found


def _index_lines(directory: Path) -> list[bytes]:
    # raw_api_bodies.mtsv › index.1; JSON Lines, 3. Line Terminator
    # is '\n'.
    found = (directory / _INDEX).read_bytes().split(b"\n")
    if found[-1] == b"":
        found.pop()
    return found


def _documents(
    directory: Path, line: bytes
) -> tuple[dict[str, Any], _Document, _Document]:
    # raw_api_bodies.mtsv › index.1, index.4.
    entry = _object(line, "an index line")
    request, response = (_held(directory, entry.get(member)) for member in _FILES)
    return entry, request, response


def _held(directory: Path, name: Any) -> _Document:
    # raw_api_bodies.mtsv › index.1: a path that is not absolute is in
    # <dir>; index.4: a file not named, or not on disk, is not held.
    if name is None:
        return None
    if not isinstance(name, str) or isinstance(name, Number):
        raise ValueError(f"{name!r} names no file")
    try:
        data = (directory / name).read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{name}: {error.strerror}") from None
    return _object(data, name)


def _object(data: bytes, what: str) -> dict[str, Any]:
    # raw_api_bodies.mtsv › index.4; index.5: of members sharing a name,
    # decode reads the last.
    try:
        value = decode(data.decode("utf-8"))
    except ValueError:
        raise ValueError(f"{what} is not JSON") from None
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return value


def _logs_data(
    entry: dict[str, Any], request: _Document, response: _Document
) -> dict[str, Any]:
    # raw_api_bodies.mtsv › event.1.
    attributes = _index_attributes(entry) + messages.attributes(request, response)
    log_record = {"eventName": _EVENT, "attributes": attributes}
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": [log_record]}]}]}


def _index_attributes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    # raw_api_bodies.mtsv › index.2, event.2.
    found = []
    if entry.get("session_id") is not None:
        found.append(
            messages.attribute("gen_ai.conversation.id", entry["session_id"], "string")
        )
    return found + [
        messages.attribute(_NAMESPACE + name, value, "any")
        for name, value in entry.items()
        if name != "session_id"
    ]


def _encode(value: Any) -> str:
    # raw_api_bodies.mtsv › event.1: the line in the OTLP JSON encoding;
    # messages.mtsv › event.4: a number as the request or response
    # writes it; RFC 8259.
    if isinstance(value, Number):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_encode(v) for v in value) + "]"
    return "{" + ",".join(f"{_encode(k)}:{_encode(v)}" for k, v in value.items()) + "}"
