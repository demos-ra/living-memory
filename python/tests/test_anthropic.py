"""Tests of integrations/providers/anthropic: Claude Code's record, rule by rule."""

import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import mtsv
from living_memory.integrations.providers import anthropic

from support import SPEC

REQUEST = {
    "model": "claude-opus-5-5",
    "max_tokens": 1024,
    "temperature": 1,
    "top_k": 40,
    "top_p": 0.9,
    "stop_sequences": ["END"],
    "stream": True,
    "output_config": {"effort": "high", "format": {"type": "json_schema"}},
    "system": "Be brief.",
    "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "get", "input": {"q": 1}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "rainy"}
            ],
        },
    ],
    "tools": [
        {"name": "get", "description": "Get.", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search"},
    ],
    "metadata": {"user_id": "u"},
}
RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "Rainy."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 20,
        "output_tokens": 7,
        "output_tokens_details": {"thinking_tokens": 3},
        "service_tier": "standard",
    },
}
ENTRY = {
    "timestamp": "2026-09-24T02:01:56Z",
    "session_id": "s1",
    "query_source": "sdk",
    "model": "claude-opus-5-5",
    "request_id": "req_1",
    "message_id": "msg_1",
    "message_uuid": "m1",
    "request_file": "a.request.json",
    "response_file": "req_1.response.json",
}


def record(directory, entries, documents):
    """Write an index of entries and the named documents into a directory."""
    folder = Path(directory)
    lines = "".join(json.dumps(entry) + "\n" for entry in entries)
    (folder / "index.jsonl").write_text(lines, encoding="utf-8")
    for name, document in documents.items():
        text = document if isinstance(document, str) else json.dumps(document)
        (folder / name).write_text(text, encoding="utf-8")


def event(request=REQUEST, response=RESPONSE, entry=ENTRY):
    """The log record written for one call, and its attributes by key."""
    with tempfile.TemporaryDirectory() as directory:
        documents = {entry["request_file"]: request, entry["response_file"]: response}
        record(directory, [entry], documents)
        (data,) = anthropic.logs_data(directory)
    (log_record,) = data["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    return log_record, {a["key"]: a["value"] for a in log_record["attributes"]}


def plain(value):
    """An AnyValue as the JSON value it maps to."""
    if "kvlistValue" in value:
        pairs = value["kvlistValue"]["values"]
        return {pair["key"]: plain(pair["value"]) for pair in pairs}
    if "arrayValue" in value:
        return [plain(v) for v in value["arrayValue"]["values"]]
    if not value:
        return None
    (member,) = value.values()
    return str(member) if isinstance(member, str) else member


def parts(attributes, key):
    return [message["parts"] for message in plain(attributes[key])]


def spec():
    with files(anthropic.__package__).joinpath("anthropic.mtsv").open("rb") as f:
        return {sheet["sheet name"]: sheet for sheet in mtsv.load(f)}


class TestSpecification(unittest.TestCase):
    def test_every_rule_named_exists(self):
        sheets = spec()
        ids = {row[0] for row in sheets["Rules"]["records"]}
        for key, field, rule in sheets["Attributes"]["records"]:
            with self.subTest(key=key, field=field):
                self.assertTrue(set(rule.split("; ")) <= ids)

    def test_structured_rows_are_the_core_sheets(self):
        with SPEC.open("rb") as f:
            core = {s["sheet name"]: s for s in mtsv.load(f)}["Sheets"]["records"]
        names = {name for name, _ in core}
        for key, field, _ in spec()["Attributes"]["records"]:
            if field:
                with self.subTest(key=key, field=field):
                    self.assertTrue(
                        [key, field] in core or f"{key}.{field}" in names
                    )

    def test_attributes_written_and_absent(self):
        # event.3, event.4: each key is written by its rules or absent.
        _, attributes = event()
        for key, field, rule in spec()["Attributes"]["records"]:
            if field or "{member}" in key:
                continue
            with self.subTest(key=key):
                if rule == "event.4":
                    self.assertNotIn(key, attributes)
                else:
                    self.assertIn(key, attributes)


class TestIndex(unittest.TestCase):
    def test_1_lines_in_order_with_their_files(self):
        with tempfile.TemporaryDirectory() as directory:
            second = dict(ENTRY, session_id="s2")
            second["request_file"] = str(Path(directory) / "b.request.json")
            documents = {
                "a.request.json": REQUEST,
                "b.request.json": REQUEST,
                "req_1.response.json": RESPONSE,
            }
            record(directory, [ENTRY, second], documents)
            found = anthropic.logs_data(directory)
        ids = []
        for data in found:
            (log_record,) = data["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
            for attribute in log_record["attributes"]:
                if attribute["key"] == "gen_ai.conversation.id":
                    ids.append(attribute["value"]["stringValue"])
        self.assertEqual(ids, ["s1", "s2"])

    def test_2_session_id_and_the_rest(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["gen_ai.conversation.id"]), "s1")
        self.assertNotIn("anthropic.claude_code.session_id", attributes)
        for member in ENTRY:
            if member != "session_id":
                with self.subTest(member):
                    key = "anthropic.claude_code." + member
                    self.assertEqual(plain(attributes[key]), ENTRY[member])

    def test_4_rejected_naming_the_line(self):
        bad_request = dict(REQUEST, max_tokens=1.5)
        cases = [
            ("[1]\n", {}),
            (json.dumps(ENTRY) + "\n", {}),
            (json.dumps(ENTRY) + "\n", {"a.request.json": "[]"}),
            (json.dumps(ENTRY) + "\n", {"a.request.json": bad_request}),
            (json.dumps(ENTRY) + "\n\n", {"a.request.json": REQUEST}),
        ]
        for index, documents in cases:
            with self.subTest(index=index, documents=list(documents)):
                with tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    (folder / "index.jsonl").write_text(index, encoding="utf-8")
                    documents = dict(documents)
                    documents.setdefault("req_1.response.json", RESPONSE)
                    for name, document in documents.items():
                        text = document
                        if not isinstance(document, str):
                            text = json.dumps(document)
                        (folder / name).write_text(text, encoding="utf-8")
                    with self.assertRaises(anthropic.RecordDecodeError) as found:
                        anthropic.logs_data(directory)
                    expected = 2 if index.endswith("\n\n") else 1
                    self.assertEqual(found.exception.lineno, expected)
                    self.assertIsInstance(found.exception, ValueError)


class TestEvent(unittest.TestCase):
    def test_1_one_event_and_nothing_else(self):
        log_record, _ = event()
        self.assertEqual(set(log_record), {"eventName", "attributes"})
        self.assertEqual(
            log_record["eventName"], "gen_ai.client.inference.operation.details"
        )

    def test_2_operation_and_provider(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["gen_ai.operation.name"]), "chat")
        self.assertEqual(plain(attributes["gen_ai.provider.name"]), "anthropic")

    def test_3_absent_where_the_record_holds_no_value(self):
        _, attributes = event(request={"messages": []}, response={})
        self.assertNotIn("gen_ai.request.model", attributes)
        self.assertNotIn("gen_ai.response.finish_reasons", attributes)

    def test_5_values(self):
        request = dict(
            REQUEST,
            metadata={"i": 7, "big": 2**64, "d": 1.5, "b": True, "n": None, "a": []},
        )
        _, attributes = event(request=request)
        metadata = attributes["anthropic.request.metadata"]["kvlistValue"]["values"]
        found = {pair["key"]: pair["value"] for pair in metadata}
        self.assertEqual(found["i"], {"intValue": "7"})
        self.assertEqual(found["big"], {"stringValue": str(2**64)})
        self.assertEqual(found["d"], {"doubleValue": "1.5"})
        self.assertEqual(found["b"], {"boolValue": True})
        self.assertEqual(found["n"], {})
        self.assertEqual(found["a"], {"arrayValue": {"values": []}})
        self.assertEqual(attributes["gen_ai.request.temperature"], {"doubleValue": "1"})
        self.assertEqual(attributes["gen_ai.request.max_tokens"], {"intValue": "1024"})


class TestRequest(unittest.TestCase):
    def test_1_parameters_and_stream(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["gen_ai.request.model"]), "claude-opus-5-5")
        self.assertEqual(plain(attributes["gen_ai.request.top_k"]), "40")
        self.assertEqual(plain(attributes["gen_ai.request.stop_sequences"]), ["END"])
        self.assertEqual(attributes["gen_ai.request.stream"], {"boolValue": True})
        _, attributes = event(request=dict(REQUEST, stream=False))
        self.assertNotIn("gen_ai.request.stream", attributes)
        self.assertNotIn("anthropic.request.stream", attributes)

    def test_2_effort_and_format(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["gen_ai.request.reasoning.level"]), "high")
        self.assertEqual(plain(attributes["gen_ai.output.type"]), "json")
        self.assertEqual(
            plain(attributes["anthropic.request.output_config"]),
            {"format": {"type": "json_schema"}},
        )

    def test_3_messages(self):
        _, attributes = event()
        messages = plain(attributes["gen_ai.input.messages"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(
            messages[0]["parts"], [{"type": "text", "content": "Weather in Paris?"}]
        )

    def test_4_system(self):
        _, attributes = event()
        self.assertEqual(
            plain(attributes["gen_ai.system_instructions"]),
            [{"type": "text", "content": "Be brief."}],
        )
        request = dict(REQUEST, system=[{"type": "text", "text": "A", "x": 1}])
        _, attributes = event(request=request)
        self.assertEqual(
            plain(attributes["gen_ai.system_instructions"]),
            [{"type": "text", "content": "A", "x": "1"}],
        )

    def test_5_tools(self):
        _, attributes = event()
        self.assertEqual(
            plain(attributes["gen_ai.tool.definitions"]),
            [
                {
                    "type": "function",
                    "name": "get",
                    "description": "Get.",
                    "parameters": {"type": "object"},
                },
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        )

    def test_6_the_rest(self):
        _, attributes = event()
        self.assertEqual(
            plain(attributes["anthropic.request.metadata"]), {"user_id": "u"}
        )
        consumed = ("model", "max_tokens", "messages", "system", "tools", "stream")
        for member in consumed:
            with self.subTest(member):
                self.assertNotIn("anthropic.request." + member, attributes)


class TestResponse(unittest.TestCase):
    def test_1_id_model_finish_reasons(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["gen_ai.response.id"]), "msg_1")
        self.assertEqual(
            plain(attributes["gen_ai.response.finish_reasons"]), ["end_turn"]
        )

    def test_2_usage(self):
        _, attributes = event()
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], {"intValue": "35"})
        self.assertEqual(
            attributes["gen_ai.usage.cache_write.input_tokens"], {"intValue": "5"}
        )
        self.assertEqual(
            attributes["gen_ai.usage.cache_read.input_tokens"], {"intValue": "20"}
        )
        self.assertEqual(attributes["gen_ai.usage.output_tokens"], {"intValue": "7"})
        self.assertEqual(
            attributes["gen_ai.usage.reasoning.output_tokens"], {"intValue": "3"}
        )
        self.assertEqual(
            plain(attributes["anthropic.response.usage"]), {"service_tier": "standard"}
        )

    def test_3_one_output_message(self):
        _, attributes = event()
        self.assertEqual(
            plain(attributes["gen_ai.output.messages"]),
            [{"role": "assistant", "parts": [{"type": "text", "content": "Rainy."}]}],
        )

    def test_4_the_rest(self):
        _, attributes = event()
        self.assertEqual(plain(attributes["anthropic.response.type"]), "message")
        self.assertIsNone(plain(attributes["anthropic.response.stop_sequence"]))
        for member in ("id", "model", "role", "content", "stop_reason"):
            with self.subTest(member):
                self.assertNotIn("anthropic.response." + member, attributes)


def block_parts(*blocks):
    """The parts written for content blocks of one user message."""
    request = dict(REQUEST, messages=[{"role": "user", "content": list(blocks)}])
    _, attributes = event(request=request)
    return parts(attributes, "gen_ai.input.messages")[0]


class TestBlock(unittest.TestCase):
    def test_1_the_rest_of_a_block(self):
        cache = {"type": "ephemeral"}
        (part,) = block_parts({"type": "text", "text": "a", "cache_control": cache})
        self.assertEqual(
            part, {"type": "text", "content": "a", "cache_control": cache}
        )

    def test_2_text(self):
        self.assertEqual(
            block_parts({"type": "text", "text": "a"}),
            [{"type": "text", "content": "a"}],
        )

    def test_3_image_and_document(self):
        base64 = {"type": "base64", "media_type": "image/png", "data": "iVBO"}
        url = {"type": "url", "url": "https://x/y.pdf"}
        file = {"type": "file", "file_id": "f1"}
        text = {"type": "text", "media_type": "text/plain", "data": "plain"}
        document = {"type": "document", "source": text}
        self.assertEqual(
            block_parts(
                {"type": "image", "source": base64},
                {"type": "document", "source": url},
                {"type": "image", "source": file},
                document,
            ),
            [
                {
                    "type": "blob",
                    "modality": "image",
                    "mime_type": "image/png",
                    "content": "iVBO",
                },
                {"type": "uri", "modality": "document", "uri": "https://x/y.pdf"},
                {"type": "file", "modality": "image", "file_id": "f1"},
                document,
            ],
        )

    def test_4_thinking(self):
        block = {"type": "thinking", "thinking": "t", "signature": "s"}
        self.assertEqual(
            block_parts(block),
            [{"type": "reasoning", "content": "t", "signature": "s"}],
        )

    def test_5_tool_use(self):
        block = {"type": "tool_use", "id": "t1", "name": "get", "input": {"q": "x"}}
        self.assertEqual(
            block_parts(block),
            [{"type": "tool_call", "id": "t1", "name": "get", "arguments": {"q": "x"}}],
        )

    def test_6_tool_result(self):
        block = {"type": "tool_result", "tool_use_id": "t1", "is_error": False}
        self.assertEqual(
            block_parts(dict(block, content="r")),
            [
                {
                    "type": "tool_call_response",
                    "id": "t1",
                    "is_error": False,
                    "response": "r",
                }
            ],
        )
        self.assertEqual(block_parts(block), [block])

    def test_7_server_tool_use(self):
        block = {
            "type": "server_tool_use",
            "id": "s1",
            "name": "web_search",
            "input": {"query": "q"},
        }
        self.assertEqual(
            block_parts(block),
            [
                {
                    "type": "server_tool_call",
                    "id": "s1",
                    "name": "web_search",
                    "server_tool_call": {"type": "web_search", "input": {"query": "q"}},
                }
            ],
        )

    def test_8_server_tool_results(self):
        block = {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        self.assertEqual(
            block_parts(block),
            [
                {
                    "type": "server_tool_call_response",
                    "id": "s1",
                    "server_tool_call_response": {
                        "type": "web_search_tool_result",
                        "content": [],
                    },
                }
            ],
        )

    def test_9_any_other_block_whole(self):
        blocks = [
            {"type": "redacted_thinking", "data": "x"},
            {"type": "search_result", "source": "s", "title": "t", "content": []},
            {"type": "compaction", "content": "c"},
        ]
        self.assertEqual(block_parts(*blocks), blocks)


class TestLoad(unittest.TestCase):
    def test_the_core_reads_what_the_reader_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            documents = {"a.request.json": REQUEST, "req_1.response.json": RESPONSE}
            record(directory, [ENTRY], documents)
            sheets = {s["sheet name"]: s for s in anthropic.load(directory)}
        messages = sheets["gen_ai.input.messages"]["records"]
        self.assertEqual([row[2] for row in messages], ["user", "assistant", "user"])
        calls = sheets["gen_ai.input.messages.parts.tool_call"]["records"]
        self.assertEqual(calls[0][2:], ["t1", "get"])


if __name__ == "__main__":
    unittest.main()
