"""How Claude Code's raw API bodies of Messages API calls are read."""

__all__ = ["load", "logs_data", "RawAPIBodiesDecodeError"]

import logging
import os
from pathlib import Path
from typing import Any

from living_memory import _event, _json_lines
from living_memory._json import Number, decode_all_pairs, encode
from living_memory._otlp_json import OTLPDecodeError, loads
from living_memory.providers.anthropic import messages

# The raw API bodies are found by their index file
# (raw_api_bodies.mtsv › index.1).
_INDEX = "index.jsonl"
# The index's members other than session_id are written under Claude
# Code's own namespace (raw_api_bodies.mtsv › index.2).
_NAMESPACE = "anthropic.claude_code."
# These members of an index line name the request and response files
# (raw_api_bodies.mtsv › index.4).
_FILES = ("request_file", "response_file")

# A library names its logger after its module and attaches no handler
# (Logging HOWTO, Configuring Logging for a Library).
_logger = logging.getLogger(__name__)

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
    # One OTLP line is written for each index line, so the core's line
    # is the index's (raw_api_bodies.mtsv › conformance.1, index.4).
    lines = "".join(encode(data) + "\n" for data in logs_data(path))
    try:
        return loads(lines)
    except OTLPDecodeError as error:
        raise RawAPIBodiesDecodeError(error.msg, error.lineno) from None


def logs_data(path: str | os.PathLike[str], /) -> list[dict[str, Any]]:
    """Return one LogsData for each line of the raw API bodies' index.

    Raise RawAPIBodiesDecodeError for raw API bodies not conforming.
    """
    # Each index line is read in order, with the files it names; a file
    # not held is reported (raw_api_bodies.mtsv › index.1, index.4).
    directory = Path(path)
    try:
        index = _json_lines.lines((directory / _INDEX).read_bytes())
    except ValueError as error:
        raise RawAPIBodiesDecodeError(str(error), 1) from None
    found = []
    for number, line in enumerate(index, 1):
        try:
            entry, request, response = _documents(directory, line)
        except ValueError as error:
            raise RawAPIBodiesDecodeError(str(error), number) from None
        for member, document in zip(_FILES, (request, response)):
            if document is None:
                _logger.warning("line %d: %s not held", number, member)
        try:
            attributes = _index_attributes(entry)
            attributes += messages.attributes(request, response)
        except ValueError as error:
            raise RawAPIBodiesDecodeError(str(error), number) from None
        found.append(_event.logs_data(attributes))
    return found


def _documents(
    directory: Path, line: bytes
) -> tuple[dict[str, Any], _Document, _Document]:
    # An index line names the request and response files it reads
    # (raw_api_bodies.mtsv › index.1, index.4).
    entry = _object(line, "an index line")
    request, response = (_held(directory, entry.get(member)) for member in _FILES)
    return entry, request, response


def _held(directory: Path, name: Any) -> _Document:
    # A path that is not absolute is in the directory, and a file not
    # named, or not on disk, is not held (raw_api_bodies.mtsv › index.1,
    # index.4).
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
    # An index line and each body is a JSON object, and members sharing
    # a name are one member holding every value (raw_api_bodies.mtsv ›
    # index.4, index.5).
    try:
        value = decode_all_pairs(data.decode("utf-8"))
    except ValueError:
        raise ValueError(f"{what} is not JSON") from None
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return value


def _index_attributes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    # The session is the conversation, and every other member is Claude
    # Code's own (raw_api_bodies.mtsv › index.2, event.2).
    found = _event.typed("gen_ai.conversation.id", entry.get("session_id"), "string")
    return found + _event.namespaced(
        _NAMESPACE,
        {name: value for name, value in entry.items() if name != "session_id"},
    )
