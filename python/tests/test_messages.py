"""Tests of anthropic/messages: the Messages API, rule by rule."""

import json
import unittest
from importlib.resources import files

import mtsv
from living_memory._json import decode
from living_memory.integrations.providers.anthropic import messages

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
# A response with only what the Messages API requires of it, event.5.
BARE = {"role": "assistant", "content": []}


def as_read(value):
    """A value as a reader of JSON gives it: numbers as Number."""
    return decode(json.dumps(value))


def attributes(request=REQUEST, response=RESPONSE):
    """The attributes written for one call, by key."""
    found = messages.attributes(as_read(request), as_read(response))
    return {a["key"]: a["value"] for a in found}


def found_value(found, key):
    (value,) = [a["value"] for a in found if a["key"] == key]
    return value


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


def spec():
    with files(messages.__package__).joinpath("messages.mtsv").open("rb") as f:
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
        # event.2, event.3: each key is written by its rules or absent.
        found = attributes()
        for key, field, rule in spec()["Attributes"]["records"]:
            if field or "{member}" in key or rule == "event.6":
                continue
            with self.subTest(key=key):
                if rule == "event.3":
                    self.assertNotIn(key, found)
                else:
                    self.assertIn(key, found)


class TestEvent(unittest.TestCase):
    def test_1_operation_and_provider(self):
        found = attributes()
        self.assertEqual(plain(found["gen_ai.operation.name"]), "chat")
        self.assertEqual(plain(found["gen_ai.provider.name"]), "anthropic")

    def test_2_absent_where_the_request_and_response_hold_no_value(self):
        found = attributes(request={"messages": []}, response=BARE)
        self.assertNotIn("gen_ai.request.model", found)
        self.assertNotIn("gen_ai.response.finish_reasons", found)

    def test_2_nothing_of_what_was_not_recorded(self):
        found = messages.attributes(None, None)
        self.assertEqual(
            [a["key"] for a in found], ["gen_ai.operation.name", "gen_ai.provider.name"]
        )

    def test_4_values(self):
        request = dict(
            REQUEST,
            metadata={"i": 7, "big": 2**64, "d": 1.5, "b": True, "n": None, "a": []},
        )
        found = attributes(request=request)
        metadata = found["anthropic.request.metadata"]["kvlistValue"]["values"]
        pairs = {pair["key"]: pair["value"] for pair in metadata}
        self.assertEqual(pairs["i"], {"intValue": "7"})
        self.assertEqual(pairs["big"], {"stringValue": str(2**64)})
        self.assertEqual(pairs["d"], {"doubleValue": "1.5"})
        self.assertEqual(pairs["b"], {"boolValue": True})
        self.assertEqual(pairs["n"], {})
        self.assertEqual(pairs["a"], {"arrayValue": {"values": []}})
        self.assertEqual(found["gen_ai.request.temperature"], {"doubleValue": "1"})
        self.assertEqual(found["gen_ai.request.max_tokens"], {"intValue": "1024"})

    def test_4_an_int_as_the_decimal_string_of_its_value(self):
        for written in ("40.0", "4e1", "40"):
            with self.subTest(written):
                request = decode(f'{{"max_tokens": {written}, "messages": []}}')
                found = messages.attributes(request, as_read(BARE))
                self.assertEqual(
                    found_value(found, "gen_ai.request.max_tokens"), {"intValue": "40"}
                )

    def test_5_rejected(self):
        cases = [
            (dict(REQUEST, max_tokens=1.5), RESPONSE),
            (dict(REQUEST, max_tokens=2**63), RESPONSE),
            (REQUEST, dict(RESPONSE, usage={"output_tokens": 2**64})),
            (dict(REQUEST, messages=[{"role": "user"}]), RESPONSE),
            (dict(REQUEST, messages=[{"content": "a"}]), RESPONSE),
            (REQUEST, {"role": "assistant"}),
            (REQUEST, {"content": []}),
            (dict(REQUEST, system=None), RESPONSE),
        ]
        for request, response in cases:
            with self.subTest(request=request, response=response):
                with self.assertRaises(ValueError):
                    messages.attributes(as_read(request), as_read(response))
        request = decode('{"temperature": 1e400, "messages": []}')
        with self.assertRaises(ValueError):
            messages.attributes(request, as_read(BARE))

    def test_6_compacted(self):
        compaction = {"type": "compaction", "content": "c"}
        cases = [
            (dict(REQUEST, messages=[{"role": "user", "content": [compaction]}]),
             RESPONSE),
            (REQUEST, dict(RESPONSE, stop_reason="compaction")),
            (REQUEST, dict(RESPONSE, content=[compaction])),
            (REQUEST, dict(RESPONSE, usage={"iterations": [{"type": "compaction"}]})),
        ]
        for request, response in cases:
            with self.subTest(request=request, response=response):
                found = attributes(request=request, response=response)
                self.assertEqual(
                    found["gen_ai.conversation.compacted"], {"boolValue": True}
                )
        self.assertNotIn("gen_ai.conversation.compacted", attributes())


class TestRequest(unittest.TestCase):
    def test_1_parameters_and_stream(self):
        found = attributes()
        self.assertEqual(plain(found["gen_ai.request.model"]), "claude-opus-5-5")
        self.assertEqual(plain(found["gen_ai.request.top_k"]), "40")
        self.assertEqual(plain(found["gen_ai.request.stop_sequences"]), ["END"])
        self.assertEqual(found["gen_ai.request.stream"], {"boolValue": True})
        found = attributes(request=dict(REQUEST, stream=False))
        self.assertNotIn("gen_ai.request.stream", found)
        self.assertNotIn("anthropic.request.stream", found)

    def test_2_effort_and_format(self):
        found = attributes()
        self.assertEqual(plain(found["gen_ai.request.reasoning.level"]), "high")
        self.assertEqual(plain(found["gen_ai.output.type"]), "json")
        self.assertEqual(
            plain(found["anthropic.request.output_config"]),
            {"format": {"type": "json_schema"}},
        )

    def test_3_messages(self):
        found = plain(attributes()["gen_ai.input.messages"])
        self.assertEqual([m["role"] for m in found], ["user", "assistant", "user"])
        self.assertEqual(
            found[0]["parts"], [{"type": "text", "content": "Weather in Paris?"}]
        )

    def test_4_system(self):
        self.assertEqual(
            plain(attributes()["gen_ai.system_instructions"]),
            [{"type": "text", "content": "Be brief."}],
        )
        request = dict(REQUEST, system=[{"type": "text", "text": "A", "x": 1}])
        self.assertEqual(
            plain(attributes(request=request)["gen_ai.system_instructions"]),
            [{"type": "text", "content": "A", "x": "1"}],
        )

    def test_5_tools(self):
        request = dict(REQUEST, tools=REQUEST["tools"] + [{"description": "d"}])
        self.assertEqual(
            plain(attributes(request=request)["gen_ai.tool.definitions"]),
            [
                {
                    "type": "function",
                    "name": "get",
                    "description": "Get.",
                    "parameters": {"type": "object"},
                },
                {"type": "web_search_20250305", "name": "web_search"},
                {"description": "d"},
            ],
        )

    def test_6_the_rest(self):
        found = attributes()
        self.assertEqual(plain(found["anthropic.request.metadata"]), {"user_id": "u"})
        consumed = ("model", "max_tokens", "messages", "system", "tools", "stream")
        for member in consumed:
            with self.subTest(member):
                self.assertNotIn("anthropic.request." + member, found)


class TestResponse(unittest.TestCase):
    def test_1_id_model_finish_reasons(self):
        found = attributes()
        self.assertEqual(plain(found["gen_ai.response.id"]), "msg_1")
        self.assertEqual(plain(found["gen_ai.response.finish_reasons"]), ["end_turn"])

    def test_2_usage(self):
        found = attributes()
        self.assertEqual(found["gen_ai.usage.input_tokens"], {"intValue": "35"})
        self.assertEqual(
            found["gen_ai.usage.cache_write.input_tokens"], {"intValue": "5"}
        )
        self.assertEqual(
            found["gen_ai.usage.cache_read.input_tokens"], {"intValue": "20"}
        )
        self.assertEqual(found["gen_ai.usage.output_tokens"], {"intValue": "7"})
        self.assertEqual(
            found["gen_ai.usage.reasoning.output_tokens"], {"intValue": "3"}
        )
        self.assertEqual(
            plain(found["anthropic.response.usage"]), {"service_tier": "standard"}
        )

    def test_3_one_output_message(self):
        self.assertEqual(
            plain(attributes()["gen_ai.output.messages"]),
            [{"role": "assistant", "parts": [{"type": "text", "content": "Rainy."}]}],
        )

    def test_4_the_rest(self):
        found = attributes()
        self.assertEqual(plain(found["anthropic.response.type"]), "message")
        self.assertIsNone(plain(found["anthropic.response.stop_sequence"]))
        for member in ("id", "model", "role", "content", "stop_reason"):
            with self.subTest(member):
                self.assertNotIn("anthropic.response." + member, found)


def block_parts(*blocks):
    """The parts written for content blocks of one user message."""
    request = dict(REQUEST, messages=[{"role": "user", "content": list(blocks)}])
    found = plain(attributes(request=request)["gen_ai.input.messages"])
    return found[0]["parts"]


class TestBlock(unittest.TestCase):
    def test_1_the_rest_of_a_block(self):
        cache = {"type": "ephemeral"}
        (part,) = block_parts({"type": "text", "text": "a", "cache_control": cache})
        self.assertEqual(part, {"type": "text", "content": "a", "cache_control": cache})

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


if __name__ == "__main__":
    unittest.main()
