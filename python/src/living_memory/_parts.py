"""The message part types, shared by the schemas that repeat them.

The definitions of OTEL-GENAI model/gen-ai/gen-ai-input-messages.json,
gen-ai-output-messages.json and gen-ai-system-instructions.json.

Constants:
TEXT -- TextPart
TOOL_CALL -- ToolCallRequestPart
TOOL_CALL_RESPONSE -- ToolCallResponsePart
SERVER_TOOL_CALL -- ServerToolCallPart
SERVER_TOOL_CALL_RESPONSE -- ServerToolCallResponsePart
BLOB -- BlobPart
FILE -- FilePart
URI -- UriPart
REASONING -- ReasoningPart
COMPACTION -- CompactionPart
GENERIC -- GenericPart
MESSAGE_PARTS -- the parts of a message, by type value, in schema order
"""

__all__ = [
    "TEXT",
    "TOOL_CALL",
    "TOOL_CALL_RESPONSE",
    "SERVER_TOOL_CALL",
    "SERVER_TOOL_CALL_RESPONSE",
    "BLOB",
    "FILE",
    "URI",
    "REASONING",
    "COMPACTION",
    "GENERIC",
    "MESSAGE_PARTS",
]

from living_memory._relations import Shape

TEXT = Shape(
    columns=("content",),
    lines=frozenset({"content"}),
    known=frozenset({"type", "content"}),
)
TOOL_CALL = Shape(
    columns=("id", "name"),
    lines=frozenset({"id", "name"}),
    nodes=("arguments",),
    known=frozenset({"type", "id", "name", "arguments"}),
)
TOOL_CALL_RESPONSE = Shape(
    columns=("id",),
    lines=frozenset({"id"}),
    nodes=("response",),
    known=frozenset({"type", "id", "response"}),
)
# GenericServerToolCall and GenericServerToolCallResponse.
_SERVER_TOOL_DETAILS = Shape(
    columns=("type",),
    lines=frozenset({"type"}),
    known=frozenset({"type"}),
)
SERVER_TOOL_CALL = Shape(
    columns=("id", "name"),
    lines=frozenset({"id", "name"}),
    children=(("server_tool_call", _SERVER_TOOL_DETAILS),),
    known=frozenset({"type", "id", "name", "server_tool_call"}),
)
SERVER_TOOL_CALL_RESPONSE = Shape(
    columns=("id",),
    lines=frozenset({"id"}),
    children=(("server_tool_call_response", _SERVER_TOOL_DETAILS),),
    known=frozenset({"type", "id", "server_tool_call_response"}),
)
BLOB = Shape(
    columns=("mime_type", "modality", "content"),
    lines=frozenset({"mime_type", "modality", "content"}),
    known=frozenset({"type", "mime_type", "modality", "content"}),
)
FILE = Shape(
    columns=("mime_type", "modality", "file_id"),
    lines=frozenset({"mime_type", "modality", "file_id"}),
    known=frozenset({"type", "mime_type", "modality", "file_id"}),
)
URI = Shape(
    columns=("mime_type", "modality", "uri"),
    lines=frozenset({"mime_type", "modality", "uri"}),
    known=frozenset({"type", "mime_type", "modality", "uri"}),
)
REASONING = Shape(
    columns=("content",),
    lines=frozenset({"content"}),
    known=frozenset({"type", "content"}),
)
COMPACTION = Shape(
    columns=("id", "content"),
    lines=frozenset({"id", "content"}),
    known=frozenset({"type", "id", "content"}),
)
GENERIC = Shape(
    columns=("type",),
    lines=frozenset({"type"}),
    known=frozenset({"type"}),
)

MESSAGE_PARTS = {
    "text": TEXT,
    "tool_call": TOOL_CALL,
    "tool_call_response": TOOL_CALL_RESPONSE,
    "server_tool_call": SERVER_TOOL_CALL,
    "server_tool_call_response": SERVER_TOOL_CALL_RESPONSE,
    "blob": BLOB,
    "file": FILE,
    "uri": URI,
    "reasoning": REASONING,
    "compaction": COMPACTION,
    "generic": GENERIC,
}
