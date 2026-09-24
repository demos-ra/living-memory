"""How Claude Code's record of Messages API calls is read."""

__all__ = ["load", "logs_data", "RecordDecodeError"]

import json
import math
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from living_memory._json import Number, decode
from living_memory.integrations import otlp_json

# anthropic.mtsv › index.1.
_INDEX = "index.jsonl"
# anthropic.mtsv › event.1.
_EVENT = "gen_ai.client.inference.operation.details"
# anthropic.mtsv › event.2.
_OPERATION = "chat"
_PROVIDER = "anthropic"
# anthropic.mtsv › index.2, request.6, response.4.
_INDEX_NAMESPACE = "anthropic.claude_code."
_REQUEST_NAMESPACE = "anthropic.request."
_RESPONSE_NAMESPACE = "anthropic.response."
# anthropic.mtsv › event.5: OTEL-COMMON, Integer Values.
_INT64 = (-(2**63), 2**63 - 1)

# anthropic.mtsv › request.1: member, attribute and Value Type.
_REQUEST = (
    ("model", "gen_ai.request.model", "string"),
    ("max_tokens", "gen_ai.request.max_tokens", "int"),
    ("temperature", "gen_ai.request.temperature", "double"),
    ("top_k", "gen_ai.request.top_k", "int"),
    ("top_p", "gen_ai.request.top_p", "double"),
    ("stop_sequences", "gen_ai.request.stop_sequences", "string[]"),
)
# anthropic.mtsv › response.1.
_RESPONSE = (
    ("id", "gen_ai.response.id", "string"),
    ("model", "gen_ai.response.model", "string"),
)
# anthropic.mtsv › response.2.
_USAGE = (
    ("cache_read_input_tokens", "gen_ai.usage.cache_read.input_tokens"),
    ("cache_creation_input_tokens", "gen_ai.usage.cache_write.input_tokens"),
    ("output_tokens", "gen_ai.usage.output_tokens"),
)
_INPUT_TOKENS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
# anthropic.mtsv › block.8.
_SERVER_TOOL_RESULTS = frozenset(
    {
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
    }
)
# anthropic.mtsv › block.3: source type, part type, member taken, and
# the part's name for it.
_SOURCES = {
    "base64": ("blob", "data", "content"),
    "url": ("uri", "url", "uri"),
    "file": ("file", "file_id", "file_id"),
}

_Attributes = list[dict[str, Any]]


class RecordDecodeError(ValueError):
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
    """Read MTSV sheets from the directory Claude Code writes its record in.

    Raise RecordDecodeError for a record that does not conform.
    """
    # anthropic.mtsv › conformance.1.
    lines = "".join(_encode(data) + "\n" for data in logs_data(path))
    return otlp_json.loads(lines)


def logs_data(path: str | os.PathLike[str], /) -> list[dict[str, Any]]:
    """Return one LogsData for each line of the record's index.

    Raise RecordDecodeError for a record that does not conform.
    """
    # anthropic.mtsv › index.1, index.4.
    directory = Path(path)
    found = []
    for index, line in enumerate(_index_lines(directory)):
        try:
            found.append(_logs_data(*_documents(directory, line)))
        except ValueError as error:
            raise RecordDecodeError(str(error), index + 1) from None
    return found


def _index_lines(directory: Path) -> list[bytes]:
    # anthropic.mtsv › index.1; JSON Lines, 3. Line Terminator is '\n'.
    found = (directory / _INDEX).read_bytes().split(b"\n")
    if found[-1] == b"":
        found.pop()
    return found


def _documents(
    directory: Path, line: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # anthropic.mtsv › index.1, index.4.
    entry = _object(line, "an index line")
    request = _object(_file(directory, entry.get("request_file")), "a request")
    response = _object(_file(directory, entry.get("response_file")), "a response")
    return entry, request, response


def _file(directory: Path, name: Any) -> bytes:
    # anthropic.mtsv › index.1: a path that is not absolute is in <dir>.
    if not isinstance(name, str):
        raise ValueError(f"{name!r} names no file")
    try:
        return (directory / name).read_bytes()
    except OSError as error:
        raise ValueError(f"{name}: {error.strerror}") from None


def _object(data: bytes, what: str) -> dict[str, Any]:
    # anthropic.mtsv › index.4.
    try:
        value = decode(data.decode("utf-8"))
    except ValueError:
        raise ValueError(f"{what} is not JSON") from None
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return value


def _logs_data(
    entry: dict[str, Any], request: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    # anthropic.mtsv › event.1, event.2.
    attributes = [
        _attribute("gen_ai.operation.name", _OPERATION, "string"),
        _attribute("gen_ai.provider.name", _PROVIDER, "string"),
    ]
    attributes += _index_attributes(entry)
    attributes += _request_attributes(request)
    attributes += _response_attributes(response)
    record = {"eventName": _EVENT, "attributes": attributes}
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}


def _index_attributes(entry: dict[str, Any]) -> _Attributes:
    # anthropic.mtsv › index.2.
    found = _typed("gen_ai.conversation.id", entry.get("session_id"), "string")
    rest = {name: value for name, value in entry.items() if name != "session_id"}
    return found + _namespaced(_INDEX_NAMESPACE, rest)


def _request_attributes(request: dict[str, Any]) -> _Attributes:
    # anthropic.mtsv › request.1 to request.6.
    found: _Attributes = []
    for member, key, kind in _REQUEST:
        found += _typed(key, request.get(member), kind)
    found += _stream(request.get("stream"))
    rest, output_config = _output_config(request.get("output_config"))
    found += output_config
    found += _typed("gen_ai.input.messages", _messages(request), "any")
    found += _typed("gen_ai.system_instructions", _system(request), "any")
    found += _typed("gen_ai.tool.definitions", _tools(request), "any")
    consumed = {member for member, _, _ in _REQUEST}
    consumed |= {"stream", "output_config", "messages", "system", "tools"}
    others = {name: value for name, value in request.items() if name not in consumed}
    if rest is not None:
        others["output_config"] = rest
    return found + _namespaced(_REQUEST_NAMESPACE, others)


def _stream(value: Any) -> _Attributes:
    # anthropic.mtsv › request.1: only when the request is streaming.
    if value is None or value is False:
        return []
    return _typed("gen_ai.request.stream", value, "boolean")


def _output_config(value: Any) -> tuple[Any, _Attributes]:
    # anthropic.mtsv › request.2: the rest of output_config, and what it
    # writes; the rest is None when nothing is left.
    if not isinstance(value, dict):
        return value, []
    found = _typed("gen_ai.request.reasoning.level", value.get("effort"), "string")
    if value.get("format") is not None:
        found += _typed("gen_ai.output.type", "json", "string")
    rest = {name: member for name, member in value.items() if name != "effort"}
    return (rest or None), found


def _messages(request: dict[str, Any]) -> Any:
    # anthropic.mtsv › request.3.
    if "messages" not in request:
        return None
    return [_message(message) for message in _array(request["messages"], "messages")]


def _message(message: Any) -> dict[str, Any]:
    # anthropic.mtsv › request.3: the role as written, the rest its own.
    if not isinstance(message, dict):
        raise ValueError("a message is not an object")
    chat = {name: value for name, value in message.items() if name != "content"}
    chat["parts"] = _parts(message.get("content"))
    return chat


def _system(request: dict[str, Any]) -> Any:
    # anthropic.mtsv › request.4.
    if "system" not in request:
        return None
    return _parts(request["system"])


def _parts(content: Any) -> list[Any]:
    # anthropic.mtsv › request.3, request.4, response.3.
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    return [_part(block) for block in _array(content, "content")]


def _tools(request: dict[str, Any]) -> Any:
    # anthropic.mtsv › request.5.
    if "tools" not in request:
        return None
    return [_tool(tool) for tool in _array(request["tools"], "tools")]


def _tool(tool: Any) -> Any:
    # anthropic.mtsv › request.5: a client tool is a function; any other
    # tool is written as the request holds it.
    client = isinstance(tool, dict) and tool.get("type", "custom") == "custom"
    if not client or "name" not in tool:
        return tool
    return _renamed(tool, "function", {"input_schema": "parameters"})


def _response_attributes(response: dict[str, Any]) -> _Attributes:
    # anthropic.mtsv › response.1 to response.4.
    found: _Attributes = []
    for member, key, kind in _RESPONSE:
        found += _typed(key, response.get(member), kind)
    stop_reason = response.get("stop_reason")
    if stop_reason is not None:
        found += _typed("gen_ai.response.finish_reasons", [stop_reason], "string[]")
    rest, usage = _usage(response.get("usage"))
    found += usage
    found += _typed("gen_ai.output.messages", _output_messages(response), "any")
    consumed = {member for member, _, _ in _RESPONSE}
    consumed |= {"stop_reason", "usage", "content", "role"}
    others = {name: value for name, value in response.items() if name not in consumed}
    if rest is not None:
        others["usage"] = rest
    return found + _namespaced(_RESPONSE_NAMESPACE, others)


def _usage(value: Any) -> tuple[Any, _Attributes]:
    # anthropic.mtsv › response.2: the rest of usage, and what it
    # writes; the rest is None when nothing is left.
    if not isinstance(value, dict):
        return value, []
    counts = [value[name] for name in _INPUT_TOKENS if value.get(name) is not None]
    found = []
    if counts:
        total = sum(_integer(count) for count in counts)
        found += _typed("gen_ai.usage.input_tokens", Number(str(total)), "int")
    for member, key in _USAGE:
        found += _typed(key, value.get(member), "int")
    rest = {name: member for name, member in value.items() if name not in _INPUT_TOKENS}
    rest.pop("output_tokens", None)
    details = rest.get("output_tokens_details")
    if isinstance(details, dict) and "thinking_tokens" in details:
        found += _typed(
            "gen_ai.usage.reasoning.output_tokens", details["thinking_tokens"], "int"
        )
        details = {n: m for n, m in details.items() if n != "thinking_tokens"}
        rest["output_tokens_details"] = details
        if not details:
            del rest["output_tokens_details"]
    return (rest or None), found


def _output_messages(response: dict[str, Any]) -> Any:
    # anthropic.mtsv › response.3: one output message.
    if "content" not in response:
        return None
    message: dict[str, Any] = {}
    if "role" in response:
        message["role"] = response["role"]
    message["parts"] = _parts(response["content"])
    return [message]


def _part(block: Any) -> Any:
    # anthropic.mtsv › block.1 to block.9.
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind == "text" and "text" in block:
        return _renamed(block, "text", {"text": "content"})
    if kind in ("image", "document"):
        return _media(block, kind)
    if kind == "thinking" and "thinking" in block:
        return _renamed(block, "reasoning", {"thinking": "content"})
    if kind == "tool_use" and "name" in block:
        return _renamed(block, "tool_call", {"input": "arguments"})
    if kind == "tool_result" and "content" in block:
        return _renamed(
            block, "tool_call_response", {"tool_use_id": "id", "content": "response"}
        )
    if kind == "server_tool_use" and "name" in block:
        return _server_tool_call(block)
    if kind in _SERVER_TOOL_RESULTS:
        return _server_tool_call_response(block)
    return block


def _renamed(
    block: dict[str, Any], kind: str, names: dict[str, str]
) -> dict[str, Any]:
    # anthropic.mtsv › block.1: consumed members under the part's names,
    # the rest as the part's own members.
    part: dict[str, Any] = {"type": kind}
    for name, value in block.items():
        if name != "type":
            part[names.get(name, name)] = value
    return part


def _media(block: dict[str, Any], kind: str) -> Any:
    # anthropic.mtsv › block.3.
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") not in _SOURCES:
        return block
    part_type, member, name = _SOURCES[source["type"]]
    if member not in source:
        return block
    part: dict[str, Any] = {"type": part_type, "modality": kind}
    if part_type == "blob" and "media_type" in source:
        part["mime_type"] = source["media_type"]
    part[name] = source[member]
    consumed = {"type", "media_type", member}
    rest = {n: value for n, value in source.items() if n not in consumed}
    for n, value in block.items():
        if n not in ("type", "source"):
            part[n] = value
    if rest:
        part["source"] = rest
    return part


def _server_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    # anthropic.mtsv › block.7.
    details = {"type": block["name"]}
    if "input" in block:
        details["input"] = block["input"]
    part = _renamed(block, "server_tool_call", {})
    part.pop("input", None)
    part["server_tool_call"] = details
    return part


def _server_tool_call_response(block: dict[str, Any]) -> dict[str, Any]:
    # anthropic.mtsv › block.8.
    details = {"type": block["type"]}
    if "content" in block:
        details["content"] = block["content"]
    part = _renamed(block, "server_tool_call_response", {"tool_use_id": "id"})
    part.pop("content", None)
    part["server_tool_call_response"] = details
    return part


def _typed(key: str, value: Any, kind: str) -> _Attributes:
    # anthropic.mtsv › event.3, event.5: an attribute only where the
    # record holds its value, of the Value Type its table gives.
    if value is None:
        return []
    return [_attribute(key, value, kind)]


def _attribute(key: str, value: Any, kind: str) -> dict[str, Any]:
    # anthropic.mtsv › event.5, index.4.
    if kind == "string" and isinstance(value, str) and not isinstance(value, Number):
        return {"key": key, "value": {"stringValue": value}}
    if kind == "boolean" and isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if kind == "int" and isinstance(value, Number) and _is_integer(value):
        return {"key": key, "value": _any_value(value)}
    if kind == "double" and isinstance(value, Number):
        return {"key": key, "value": {"doubleValue": value}}
    if kind == "string[]" and _strings(value):
        return {"key": key, "value": _any_value(value)}
    if kind == "any":
        return {"key": key, "value": _any_value(value)}
    raise ValueError(f"{key} is not a {kind}")


def _namespaced(namespace: str, members: dict[str, Any]) -> _Attributes:
    # anthropic.mtsv › index.2, request.6, response.4.
    return [
        _attribute(namespace + name, value, "any")
        for name, value in members.items()
    ]


def _any_value(value: Any) -> dict[str, Any]:
    # anthropic.mtsv › event.5; OTEL-COMMON, Converting to AnyValue.
    if isinstance(value, dict):
        pairs = [{"key": name, "value": _any_value(v)} for name, v in value.items()]
        return {"kvlistValue": {"values": pairs}}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(v) for v in value]}}
    if isinstance(value, bool):
        return {"boolValue": value}
    if value is None:
        return {}
    if isinstance(value, Number):
        return _number(value)
    return {"stringValue": value}


def _number(value: Number) -> dict[str, Any]:
    # anthropic.mtsv › event.5; OTEL-COMMON, Integer Values and Floating
    # Point Values; JSON-SCHEMA-07 validation, 6.1.1.
    if _is_integer(value):
        low, high = _INT64
        if low <= Decimal(value) <= high:
            return {"intValue": str(value)}
        return {"stringValue": str(value)}
    if math.isinf(float(value)):
        return {"stringValue": str(value)}
    return {"doubleValue": value}


def _integer(value: Any) -> int:
    # anthropic.mtsv › response.2, index.4.
    if not isinstance(value, Number) or not _is_integer(value):
        raise ValueError(f"{value!r} is not an integer")
    return int(Decimal(value))


def _is_integer(value: Number) -> bool:
    number = Decimal(value)
    return number == number.to_integral()


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(v, str) and not isinstance(v, Number) for v in value
    )


def _array(value: Any, name: str) -> list[Any]:
    # anthropic.mtsv › index.4.
    if not isinstance(value, list):
        raise ValueError(f"{name} is not an array")
    return value


def _encode(value: Any) -> str:
    # anthropic.mtsv › event.5: a number as the record writes it;
    # RFC 8259.
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
