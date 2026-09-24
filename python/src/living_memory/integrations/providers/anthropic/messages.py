"""How a Messages API call becomes the attributes of an event."""

__all__ = ["attribute", "attributes"]

import math
from decimal import Decimal
from typing import Any

from living_memory._json import Number

# messages.mtsv › event.1.
_OPERATION = "chat"
_PROVIDER = "anthropic"
# messages.mtsv › request.6, response.4.
_REQUEST_NAMESPACE = "anthropic.request."
_RESPONSE_NAMESPACE = "anthropic.response."
# messages.mtsv › event.4: OTEL-COMMON, Integer Values.
_INT64 = (-(2**63), 2**63 - 1)
# messages.mtsv › event.6.
_COMPACTION = "compaction"

# messages.mtsv › request.1: member, attribute and Value Type.
_REQUEST = (
    ("model", "gen_ai.request.model", "string"),
    ("max_tokens", "gen_ai.request.max_tokens", "int"),
    ("temperature", "gen_ai.request.temperature", "double"),
    ("top_k", "gen_ai.request.top_k", "int"),
    ("top_p", "gen_ai.request.top_p", "double"),
    ("stop_sequences", "gen_ai.request.stop_sequences", "string[]"),
)
# messages.mtsv › response.1.
_RESPONSE = (
    ("id", "gen_ai.response.id", "string"),
    ("model", "gen_ai.response.model", "string"),
)
# messages.mtsv › response.2.
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
# messages.mtsv › block.8.
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
# messages.mtsv › block.3: source type, part type, member taken, and
# the part's name for it.
_SOURCES = {
    "base64": ("blob", "data", "content"),
    "url": ("uri", "url", "uri"),
    "file": ("file", "file_id", "file_id"),
}

_Attributes = list[dict[str, Any]]
_Document = dict[str, Any] | None


def attributes(request: _Document, response: _Document) -> _Attributes:
    """Return the event's attributes for a request and its response.

    A request or response that was not recorded is None.
    Raise ValueError for a request or response that does not conform.
    """
    # messages.mtsv › conformance.1, event.1; event.2: nothing of what
    # was not recorded.
    found = [
        attribute("gen_ai.operation.name", _OPERATION, "string"),
        attribute("gen_ai.provider.name", _PROVIDER, "string"),
    ]
    if request is not None:
        found += _request_attributes(request)
    if response is not None:
        found += _response_attributes(response)
    if _compacted(request, response):
        found.append(attribute("gen_ai.conversation.compacted", True, "boolean"))
    return found


def attribute(key: str, value: Any, kind: str) -> dict[str, Any]:
    """Return an attribute of a Value Type, any for an AnyValue.

    Raise ValueError for a value that is not of that type.
    """
    # messages.mtsv › event.4, event.5.
    if kind == "string" and isinstance(value, str) and not isinstance(value, Number):
        return {"key": key, "value": {"stringValue": value}}
    if kind == "boolean" and isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if kind == "int" and _is_int64(value):
        return {"key": key, "value": _any_value(value)}
    if kind == "double" and _is_double(value):
        return {"key": key, "value": {"doubleValue": value}}
    if kind == "string[]" and _strings(value):
        return {"key": key, "value": _any_value(value)}
    if kind == "any":
        return {"key": key, "value": _any_value(value)}
    raise ValueError(f"{key} is not a {kind}")


def _request_attributes(request: dict[str, Any]) -> _Attributes:
    # messages.mtsv › request.1 to request.6.
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
    # messages.mtsv › request.1: only when the request is streaming.
    if value is None or value is False:
        return []
    return _typed("gen_ai.request.stream", value, "boolean")


def _output_config(value: Any) -> tuple[Any, _Attributes]:
    # messages.mtsv › request.2: the rest of output_config, and what it
    # writes; the rest is None when nothing is left.
    if not isinstance(value, dict):
        return value, []
    found = _typed("gen_ai.request.reasoning.level", value.get("effort"), "string")
    if value.get("format") is not None:
        found += _typed("gen_ai.output.type", "json", "string")
    rest = {name: member for name, member in value.items() if name != "effort"}
    return (rest or None), found


def _messages(request: dict[str, Any]) -> Any:
    # messages.mtsv › request.3.
    if "messages" not in request:
        return None
    return [_message(message) for message in _array(request["messages"], "messages")]


def _message(message: Any) -> dict[str, Any]:
    # messages.mtsv › request.3, event.5: a role and a content; the role
    # as written, the rest its own.
    if not isinstance(message, dict):
        raise ValueError("a message is not an object")
    for member in ("role", "content"):
        if member not in message:
            raise ValueError(f"a message has no {member}")
    chat = {name: value for name, value in message.items() if name != "content"}
    chat["parts"] = _parts(message["content"])
    return chat


def _system(request: dict[str, Any]) -> Any:
    # messages.mtsv › request.4.
    if "system" not in request:
        return None
    return _parts(request["system"])


def _parts(content: Any) -> list[Any]:
    # messages.mtsv › request.3, request.4, response.3.
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    return [_part(block) for block in _array(content, "content")]


def _tools(request: dict[str, Any]) -> Any:
    # messages.mtsv › request.5.
    if "tools" not in request:
        return None
    return [_tool(tool) for tool in _array(request["tools"], "tools")]


def _tool(tool: Any) -> Any:
    # messages.mtsv › request.5: a client tool with a name is a
    # function; any other tool is written as the request holds it.
    client = isinstance(tool, dict) and tool.get("type", "custom") == "custom"
    if not client or "name" not in tool:
        return tool
    return _renamed(tool, "function", {"input_schema": "parameters"})


def _response_attributes(response: dict[str, Any]) -> _Attributes:
    # messages.mtsv › response.1 to response.4.
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
    # messages.mtsv › response.2: the rest of usage, and what it
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


def _output_messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    # messages.mtsv › response.3, event.5: one output message, from a
    # response with a role and a content.
    for member in ("role", "content"):
        if member not in response:
            raise ValueError(f"a response has no {member}")
    return [{"role": response["role"], "parts": _parts(response["content"])}]


def _compacted(request: _Document, response: _Document) -> bool:
    # messages.mtsv › event.6: a compaction block in a message of the
    # request or in the response, a stop_reason of compaction, or an
    # iteration of that type in usage.
    blocks = []
    for message in (request or {}).get("messages", []):
        if isinstance(message["content"], list):
            blocks += message["content"]
    response = response or {}
    if isinstance(response.get("content"), list):
        blocks += response["content"]
    usage = response.get("usage")
    iterations = usage.get("iterations") if isinstance(usage, dict) else None
    if isinstance(iterations, list):
        blocks += iterations
    kinds = {block.get("type") for block in blocks if isinstance(block, dict)}
    return _COMPACTION in kinds or response.get("stop_reason") == _COMPACTION


def _part(block: Any) -> Any:
    # messages.mtsv › block.1 to block.9.
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
    # messages.mtsv › block.1: consumed members under the part's names,
    # the rest as the part's own members.
    part: dict[str, Any] = {"type": kind}
    for name, value in block.items():
        if name != "type":
            part[names.get(name, name)] = value
    return part


def _media(block: dict[str, Any], kind: str) -> Any:
    # messages.mtsv › block.3.
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
    # messages.mtsv › block.7.
    details = {"type": block["name"]}
    if "input" in block:
        details["input"] = block["input"]
    part = _renamed(block, "server_tool_call", {})
    part.pop("input", None)
    part["server_tool_call"] = details
    return part


def _server_tool_call_response(block: dict[str, Any]) -> dict[str, Any]:
    # messages.mtsv › block.8.
    details = {"type": block["type"]}
    if "content" in block:
        details["content"] = block["content"]
    part = _renamed(block, "server_tool_call_response", {"tool_use_id": "id"})
    part.pop("content", None)
    part["server_tool_call_response"] = details
    return part


def _typed(key: str, value: Any, kind: str) -> _Attributes:
    # messages.mtsv › event.2, event.4: an attribute only where the
    # request and response hold its value, of the Value Type its table
    # gives.
    if value is None:
        return []
    return [attribute(key, value, kind)]


def _namespaced(namespace: str, members: dict[str, Any]) -> _Attributes:
    # messages.mtsv › request.6, response.4.
    return [
        attribute(namespace + name, value, "any") for name, value in members.items()
    ]


def _any_value(value: Any) -> dict[str, Any]:
    # messages.mtsv › event.4; OTEL-COMMON, Converting to AnyValue.
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
    # messages.mtsv › event.4; OTEL-COMMON, Integer Values and Floating
    # Point Values; JSON-SCHEMA-07 validation, 6.1.1.
    if _is_int64(value):
        return {"intValue": str(int(Decimal(value)))}
    if _is_integer(value) or not _is_double(value):
        return {"stringValue": str(value)}
    return {"doubleValue": value}


def _integer(value: Any) -> int:
    # messages.mtsv › response.2, event.5.
    if not isinstance(value, Number) or not _is_integer(value):
        raise ValueError(f"{value!r} is not an integer")
    return int(Decimal(value))


def _is_int64(value: Any) -> bool:
    # messages.mtsv › event.4: a zero fractional part, within the 64-bit
    # signed range.
    if not isinstance(value, Number) or not _is_integer(value):
        return False
    low, high = _INT64
    return low <= Decimal(value) <= high


def _is_double(value: Any) -> bool:
    # messages.mtsv › event.4: within the range of an IEEE 754 64-bit
    # double.
    return isinstance(value, Number) and not math.isinf(float(value))


def _is_integer(value: Number) -> bool:
    number = Decimal(value)
    return number == number.to_integral()


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(v, str) and not isinstance(v, Number) for v in value
    )


def _array(value: Any, name: str) -> list[Any]:
    # messages.mtsv › event.5.
    if not isinstance(value, list):
        raise ValueError(f"{name} is not an array")
    return value
