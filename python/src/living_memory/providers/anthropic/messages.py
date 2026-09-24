"""How a Messages API call becomes the attributes of an event."""

__all__ = ["attributes"]

from decimal import Decimal
from typing import Any

from living_memory import _any_value
from living_memory._event import attribute, namespaced, typed
from living_memory._json import Number, array

# The operation is a chat completion, and the provider is Anthropic
# (messages.mtsv › event.1).
_OPERATION = "chat"
_PROVIDER = "anthropic"
# A field with no GenAI attribute is written under Anthropic's own
# namespaces (messages.mtsv › request.6, response.4).
_REQUEST_NAMESPACE = "anthropic.request."
_RESPONSE_NAMESPACE = "anthropic.response."
# Compaction is shown by a block, a stop reason or a usage iteration
# of this type (messages.mtsv › event.6).
_COMPACTION = "compaction"

# These request members are written as GenAI attributes, each with
# its Value Type (messages.mtsv › request.1).
_REQUEST = (
    ("model", "gen_ai.request.model", "string"),
    ("max_tokens", "gen_ai.request.max_tokens", "int"),
    ("temperature", "gen_ai.request.temperature", "double"),
    ("top_k", "gen_ai.request.top_k", "int"),
    ("top_p", "gen_ai.request.top_p", "double"),
    ("stop_sequences", "gen_ai.request.stop_sequences", "string[]"),
)
# These response members are written as GenAI attributes
# (messages.mtsv › response.1).
_RESPONSE = (
    ("id", "gen_ai.response.id", "string"),
    ("model", "gen_ai.response.model", "string"),
)
# These usage members are written as GenAI attributes, and the input
# tokens are the sum of the three counts (messages.mtsv › response.2).
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
# These blocks are server tool results (messages.mtsv › block.8).
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
# Each source type gives the part it becomes, the member taken from the
# source, and the part's name for that member (messages.mtsv ›
# block.3).
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
    # The operation and provider are always written, and nothing of what
    # was not recorded (messages.mtsv › event.1, event.2).
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


def _request_attributes(request: dict[str, Any]) -> _Attributes:
    # The request's members are written by their rules, and the rest
    # under Anthropic's namespace (messages.mtsv › request.1 to
    # request.6).
    found: _Attributes = []
    for member, key, value_type in _REQUEST:
        found += typed(key, request.get(member), value_type)
    found += _stream(request.get("stream"))
    rest, output_config = _output_config(request.get("output_config"))
    found += output_config
    found += typed("gen_ai.input.messages", _messages(request), "any")
    found += typed("gen_ai.system_instructions", _system(request), "any")
    found += typed("gen_ai.tool.definitions", _tools(request), "any")
    consumed = {member for member, _, _ in _REQUEST}
    consumed |= {"stream", "output_config", "messages", "system", "tools"}
    others = {name: value for name, value in request.items() if name not in consumed}
    if rest is not None:
        others["output_config"] = rest
    return found + namespaced(_REQUEST_NAMESPACE, others)


def _stream(value: Any) -> _Attributes:
    # The stream attribute is set only when the request is streaming
    # (messages.mtsv › request.1).
    if value is None or value is False:
        return []
    return typed("gen_ai.request.stream", value, "boolean")


def _output_config(value: Any) -> tuple[Any, _Attributes]:
    # The effort is the reasoning level, and a format asks for JSON; the
    # rest of output_config, None when nothing is left, is Anthropic's
    # own (messages.mtsv › request.2).
    if not isinstance(value, dict):
        return value, []
    found = typed("gen_ai.request.reasoning.level", value.get("effort"), "string")
    if value.get("format") is not None:
        found += typed("gen_ai.output.type", "json", "string")
    rest = {name: member for name, member in value.items() if name != "effort"}
    return (rest or None), found


def _messages(request: dict[str, Any]) -> Any:
    # The messages are the input messages, in the order sent
    # (messages.mtsv › request.3).
    if "messages" not in request:
        return None
    return [_message(message) for message in _array(request["messages"], "messages")]


def _message(message: Any) -> dict[str, Any]:
    # A message has a role and a content; the role is kept as written,
    # and the rest of the message is its own (messages.mtsv › request.3,
    # event.5).
    if not isinstance(message, dict):
        raise ValueError("a message is not an object")
    for member in ("role", "content"):
        if member not in message:
            raise ValueError(f"a message has no {member}")
    chat = {name: value for name, value in message.items() if name != "content"}
    chat["parts"] = _parts(message["content"])
    return chat


def _system(request: dict[str, Any]) -> Any:
    # The system prompt is the system instructions
    # (messages.mtsv › request.4).
    if "system" not in request:
        return None
    return _parts(request["system"])


def _parts(content: Any) -> list[Any]:
    # A string content is one text part, and an array one part per block
    # (messages.mtsv › request.3, request.4, response.3).
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    return [_part(block) for block in _array(content, "content")]


def _tools(request: dict[str, Any]) -> Any:
    # The tools are the tool definitions (messages.mtsv › request.5).
    if "tools" not in request:
        return None
    return [_tool(tool) for tool in _array(request["tools"], "tools")]


def _tool(tool: Any) -> Any:
    # A client tool with a name is a function; any other tool is written
    # as the request holds it (messages.mtsv › request.5).
    client = isinstance(tool, dict) and tool.get("type", "custom") == "custom"
    if not client or "name" not in tool:
        return tool
    return _renamed(tool, "function", {"input_schema": "parameters"})


def _response_attributes(response: dict[str, Any]) -> _Attributes:
    # The response's members are written by their rules, and the rest
    # under Anthropic's namespace (messages.mtsv › response.1 to
    # response.4).
    found: _Attributes = []
    for member, key, value_type in _RESPONSE:
        found += typed(key, response.get(member), value_type)
    stop_reason = response.get("stop_reason")
    if stop_reason is not None:
        found += typed("gen_ai.response.finish_reasons", [stop_reason], "string[]")
    rest, usage = _usage(response.get("usage"))
    found += usage
    found += typed("gen_ai.output.messages", _output_messages(response), "any")
    consumed = {member for member, _, _ in _RESPONSE}
    consumed |= {"stop_reason", "usage", "content", "role"}
    others = {name: value for name, value in response.items() if name not in consumed}
    if rest is not None:
        others["usage"] = rest
    return found + namespaced(_RESPONSE_NAMESPACE, others)


def _usage(value: Any) -> tuple[Any, _Attributes]:
    # The input tokens are the sum of the three counts, and the thinking
    # tokens the reasoning output tokens; the rest of usage, None when
    # nothing is left, is Anthropic's own (messages.mtsv › response.2).
    if not isinstance(value, dict):
        return value, []
    counts = [value[name] for name in _INPUT_TOKENS if value.get(name) is not None]
    found = []
    if counts:
        total = sum(_integer(count) for count in counts)
        found += typed("gen_ai.usage.input_tokens", Number(str(total)), "int")
    for member, key in _USAGE:
        found += typed(key, value.get(member), "int")
    rest = {name: member for name, member in value.items() if name not in _INPUT_TOKENS}
    rest.pop("output_tokens", None)
    details = rest.get("output_tokens_details")
    if isinstance(details, dict) and "thinking_tokens" in details:
        thinking = details["thinking_tokens"]
        found += typed("gen_ai.usage.reasoning.output_tokens", thinking, "int")
        details = {
            name: member
            for name, member in details.items()
            if name != "thinking_tokens"
        }
        rest["output_tokens_details"] = details
        if not details:
            del rest["output_tokens_details"]
    return (rest or None), found


def _output_messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    # A response with a role and a content is one output message
    # (messages.mtsv › response.3, event.5).
    for member in ("role", "content"):
        if member not in response:
            raise ValueError(f"a response has no {member}")
    return [{"role": response["role"], "parts": _parts(response["content"])}]


def _compacted(request: _Document, response: _Document) -> bool:
    # A compaction block in a message of the request or in the response,
    # a stop_reason of compaction, or an iteration of that type in usage
    # (messages.mtsv › event.6).
    blocks: list[Any] = []
    for message in array((request or {}).get("messages")):
        if isinstance(message, dict):
            blocks += array(message.get("content"))
    response = response or {}
    blocks += array(response.get("content"))
    usage = response.get("usage")
    if isinstance(usage, dict):
        blocks += array(usage.get("iterations"))
    kinds = {block.get("type") for block in blocks if isinstance(block, dict)}
    return _COMPACTION in kinds or response.get("stop_reason") == _COMPACTION


def _part(block: Any) -> Any:
    # A content block becomes the part a block rule names, and any other
    # block is carried whole (messages.mtsv › block.1 to block.9).
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type == "text" and "text" in block:
        return _renamed(block, "text", {"text": "content"})
    if block_type in ("image", "document"):
        return _media(block, block_type)
    if block_type == "thinking" and "thinking" in block:
        return _renamed(block, "reasoning", {"thinking": "content"})
    if block_type == "tool_use" and "name" in block:
        return _renamed(block, "tool_call", {"input": "arguments"})
    if block_type == "tool_result" and "content" in block:
        return _renamed(
            block, "tool_call_response", {"tool_use_id": "id", "content": "response"}
        )
    if block_type == "server_tool_use" and "name" in block:
        return _server_tool_call(block)
    if block_type in _SERVER_TOOL_RESULTS:
        return _server_tool_call_response(block)
    return block


def _renamed(
    block: dict[str, Any], part_type: str, names: dict[str, str]
) -> dict[str, Any]:
    # The consumed members take the part's names, and the rest of the
    # block are the part's own members (messages.mtsv › block.1).
    part: dict[str, Any] = {"type": part_type}
    for name, value in block.items():
        if name != "type":
            part[names.get(name, name)] = value
    return part


def _media(block: dict[str, Any], modality: str) -> Any:
    # An image or a document becomes a blob, uri or file part by its
    # source's type (messages.mtsv › block.3).
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") not in _SOURCES:
        return block
    part_type, member, name = _SOURCES[source["type"]]
    if member not in source:
        return block
    part: dict[str, Any] = {"type": part_type, "modality": modality}
    if part_type == "blob" and "media_type" in source:
        part["mime_type"] = source["media_type"]
    part[name] = source[member]
    consumed = {"type", "media_type", member}
    rest = {key: value for key, value in source.items() if key not in consumed}
    for key, value in block.items():
        if key not in ("type", "source"):
            part[key] = value
    if rest:
        part["source"] = rest
    return part


def _server_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    # A server tool use becomes a server tool call, its details typed by
    # the tool's name (messages.mtsv › block.7).
    details = {"type": block["name"]}
    if "input" in block:
        details["input"] = block["input"]
    part = _renamed(block, "server_tool_call", {})
    part.pop("input", None)
    part["server_tool_call"] = details
    return part


def _server_tool_call_response(block: dict[str, Any]) -> dict[str, Any]:
    # A server tool result becomes a server tool call response, its
    # details typed by the block's type (messages.mtsv › block.8).
    details = {"type": block["type"]}
    if "content" in block:
        details["content"] = block["content"]
    part = _renamed(block, "server_tool_call_response", {"tool_use_id": "id"})
    part.pop("content", None)
    part["server_tool_call_response"] = details
    return part


def _integer(value: Any) -> int:
    # A count that is not an integer is non-conforming
    # (messages.mtsv › response.2, event.5).
    if not isinstance(value, Number) or not _any_value.is_integer(value):
        raise ValueError(f"{value!r} is not an integer")
    return int(Decimal(value))


def _array(value: Any, name: str) -> list[Any]:
    # A member the Messages API holds as an array is non-conforming as
    # anything else (messages.mtsv › event.5).
    if not isinstance(value, list):
        raise ValueError(f"{name} is not an array")
    return value
