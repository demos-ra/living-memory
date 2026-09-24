"""The message part types, shared by the schemas that repeat them.

OTEL-GENAI model/gen-ai/gen-ai-input-messages.json,
gen-ai-output-messages.json and gen-ai-system-instructions.json.
"""

__all__ = [
    "belongs_to",
    "TEXT",
    "GENERIC",
    "MESSAGE_PARTS",
    "DEFS",
    "ITEMS",
    "DEFINITIONS",
]

from collections.abc import Callable
from typing import Any

from living_memory import _json_schema
from living_memory._relations import Definition, Variants

TEXT = Definition(
    columns=("content",),
    lines=frozenset({"content"}),
    properties=frozenset({"type", "content"}),
)
_TOOL_CALL = Definition(
    columns=("id", "name"),
    lines=frozenset({"id", "name"}),
    nodes=("arguments",),
    properties=frozenset({"type", "id", "name", "arguments"}),
)
_TOOL_CALL_RESPONSE = Definition(
    columns=("id",),
    lines=frozenset({"id"}),
    nodes=("response",),
    properties=frozenset({"type", "id", "response"}),
)
# GenericServerToolCall and GenericServerToolCallResponse.
_SERVER_TOOL_DETAILS = Definition(
    columns=("type",),
    lines=frozenset({"type"}),
    properties=frozenset({"type"}),
)
_SERVER_TOOL_CALL = Definition(
    columns=("id", "name"),
    lines=frozenset({"id", "name"}),
    children=(("server_tool_call", _SERVER_TOOL_DETAILS),),
    properties=frozenset({"type", "id", "name", "server_tool_call"}),
)
_SERVER_TOOL_CALL_RESPONSE = Definition(
    columns=("id",),
    lines=frozenset({"id"}),
    children=(("server_tool_call_response", _SERVER_TOOL_DETAILS),),
    properties=frozenset({"type", "id", "server_tool_call_response"}),
)
_BLOB = Definition(
    columns=("mime_type", "modality", "content"),
    lines=frozenset({"mime_type", "modality", "content"}),
    properties=frozenset({"type", "mime_type", "modality", "content"}),
)
_FILE = Definition(
    columns=("mime_type", "modality", "file_id"),
    lines=frozenset({"mime_type", "modality", "file_id"}),
    properties=frozenset({"type", "mime_type", "modality", "file_id"}),
)
_URI = Definition(
    columns=("mime_type", "modality", "uri"),
    lines=frozenset({"mime_type", "modality", "uri"}),
    properties=frozenset({"type", "mime_type", "modality", "uri"}),
)
_REASONING = Definition(
    columns=("content",),
    lines=frozenset({"content"}),
    properties=frozenset({"type", "content"}),
)
_COMPACTION = Definition(
    columns=("id", "content"),
    lines=frozenset({"id", "content"}),
    properties=frozenset({"type", "id", "content"}),
)
GENERIC = Definition(
    columns=("type",),
    lines=frozenset({"type"}),
    properties=frozenset({"type"}),
)

_MESSAGE_PARTS = {
    "text": TEXT,
    "tool_call": _TOOL_CALL,
    "tool_call_response": _TOOL_CALL_RESPONSE,
    "server_tool_call": _SERVER_TOOL_CALL,
    "server_tool_call_response": _SERVER_TOOL_CALL_RESPONSE,
    "blob": _BLOB,
    "file": _FILE,
    "uri": _URI,
    "reasoning": _REASONING,
    "compaction": _COMPACTION,
    "generic": GENERIC,
}

_STRING_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
_MODALITY = {"anyOf": [{"$ref": "#/$defs/Modality"}, {"type": "string"}]}

# OTEL-GENAI model/gen-ai/gen-ai-input-messages.json, "$defs", written
# out by hand without their title and description annotations.
DEFS: dict[str, Any] = {
    "BlobPart": {
        "properties": {
            "type": {"const": "blob", "type": "string"},
            "mime_type": _STRING_OR_NULL,
            "modality": _MODALITY,
            "content": {"format": "binary", "type": "string"},
        },
        "required": ["type", "modality", "content"],
        "type": "object",
    },
    "CompactionPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "compaction", "type": "string"},
            "id": _STRING_OR_NULL,
            "content": _STRING_OR_NULL,
        },
        "required": ["type"],
        "type": "object",
    },
    "FilePart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "file", "type": "string"},
            "mime_type": _STRING_OR_NULL,
            "modality": _MODALITY,
            "file_id": {"type": "string"},
        },
        "required": ["type", "modality", "file_id"],
        "type": "object",
    },
    "GenericPart": {
        "additionalProperties": True,
        "properties": {"type": {"type": "string"}},
        "required": ["type"],
        "type": "object",
    },
    "GenericServerToolCall": {
        "additionalProperties": True,
        "properties": {"type": {"type": "string"}},
        "required": ["type"],
        "type": "object",
    },
    "GenericServerToolCallResponse": {
        "additionalProperties": True,
        "properties": {"type": {"type": "string"}},
        "required": ["type"],
        "type": "object",
    },
    "Modality": {
        "enum": ["image", "video", "audio", "document"],
        "type": "string",
    },
    "ReasoningPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "reasoning", "type": "string"},
            "content": {"type": "string"},
        },
        "required": ["type", "content"],
        "type": "object",
    },
    "ServerToolCallPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "server_tool_call", "type": "string"},
            "id": _STRING_OR_NULL,
            "name": {"type": "string"},
            "server_tool_call": {"$ref": "#/$defs/GenericServerToolCall"},
        },
        "required": ["type", "name", "server_tool_call"],
        "type": "object",
    },
    "ServerToolCallResponsePart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "server_tool_call_response", "type": "string"},
            "id": _STRING_OR_NULL,
            "server_tool_call_response": {
                "$ref": "#/$defs/GenericServerToolCallResponse"
            },
        },
        "required": ["type", "server_tool_call_response"],
        "type": "object",
    },
    "TextPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "text", "type": "string"},
            "content": {"type": "string"},
        },
        "required": ["type", "content"],
        "type": "object",
    },
    "ToolCallRequestPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "tool_call", "type": "string"},
            "id": _STRING_OR_NULL,
            "name": {"type": "string"},
            "arguments": {"default": None},
        },
        "required": ["type", "name"],
        "type": "object",
    },
    "ToolCallResponsePart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "tool_call_response", "type": "string"},
            "id": _STRING_OR_NULL,
            "response": {},
        },
        "required": ["type", "response"],
        "type": "object",
    },
    "UriPart": {
        "additionalProperties": True,
        "properties": {
            "type": {"const": "uri", "type": "string"},
            "mime_type": _STRING_OR_NULL,
            "modality": _MODALITY,
            "uri": {"type": "string"},
        },
        "required": ["type", "modality", "uri"],
        "type": "object",
    },
}

# OTEL-GENAI gen-ai-input-messages.json, ChatMessage, parts: the items,
# as gen-ai-output-messages.json repeats them.
ITEMS = {
    "anyOf": [
        {"$ref": "#/$defs/TextPart"},
        {"$ref": "#/$defs/ToolCallRequestPart"},
        {"$ref": "#/$defs/ToolCallResponsePart"},
        {"$ref": "#/$defs/ServerToolCallPart"},
        {"$ref": "#/$defs/ServerToolCallResponsePart"},
        {"$ref": "#/$defs/BlobPart"},
        {"$ref": "#/$defs/FilePart"},
        {"$ref": "#/$defs/UriPart"},
        {"$ref": "#/$defs/ReasoningPart"},
        {"$ref": "#/$defs/CompactionPart"},
        {"$ref": "#/$defs/GenericPart"},
    ]
}

# spec › item.1.
DEFINITIONS = {
    definition["properties"]["type"]["const"]: name
    for name, definition in DEFS.items()
    if "const" in definition.get("properties", {}).get("type", {})
}


def belongs_to(
    definitions: dict[str, str], root: dict[str, Any]
) -> Callable[[str, Any], bool]:
    # spec › item.1.
    def belongs(name: str, item: Any) -> bool:
        if name not in definitions:
            return False
        reference = {"$ref": f"#/$defs/{definitions[name]}"}
        return _json_schema.validates(item, reference, root)

    return belongs


MESSAGE_PARTS = Variants(_MESSAGE_PARTS, belongs_to(DEFINITIONS, {"$defs": DEFS}))
