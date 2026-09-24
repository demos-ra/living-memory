"""Tests of claude_code/raw_api_bodies, rule by rule."""

import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import mtsv
from living_memory.integrations.providers.anthropic.claude_code import (
    raw_api_bodies,
)

from test_messages import REQUEST, RESPONSE

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


def write(directory, index, documents):
    """Write an index's text and the named documents in a directory."""
    folder = Path(directory)
    (folder / "index.jsonl").write_text(index, encoding="utf-8")
    for name, document in documents.items():
        text = document if isinstance(document, str) else json.dumps(document)
        (folder / name).write_text(text, encoding="utf-8")


def lines(*entries):
    return "".join(json.dumps(entry) + "\n" for entry in entries)


def event(request=REQUEST, response=RESPONSE, entry=ENTRY):
    """The log record written for a call, and its attributes by key."""
    documents = {entry["request_file"]: request, entry["response_file"]: response}
    with tempfile.TemporaryDirectory() as directory:
        write(directory, lines(entry), documents)
        (data,) = raw_api_bodies.logs_data(directory)
    (log_record,) = data["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    return log_record, {a["key"]: a["value"] for a in log_record["attributes"]}


def rejected(test, index, documents):
    """The error raised for raw API bodies, both read and converted."""
    documents = dict(documents)
    documents.setdefault("a.request.json", REQUEST)
    documents.setdefault("req_1.response.json", RESPONSE)
    found = []
    for read in (raw_api_bodies.logs_data, raw_api_bodies.load):
        with tempfile.TemporaryDirectory() as directory:
            write(directory, index, documents)
            with test.assertRaises(raw_api_bodies.RawAPIBodiesDecodeError) as caught:
                read(directory)
        found.append(caught.exception)
    return found


def spec():
    package = files(raw_api_bodies.__package__)
    with package.joinpath("raw_api_bodies.mtsv").open("rb") as f:
        return {sheet["sheet name"]: sheet for sheet in mtsv.load(f)}


class TestSpecification(unittest.TestCase):
    def test_every_rule_named_exists(self):
        sheets = spec()
        ids = {row[0] for row in sheets["Rules"]["records"]}
        for key, field, rule in sheets["Attributes"]["records"]:
            with self.subTest(key=key):
                self.assertTrue(set(rule.split("; ")) <= ids)

    def test_attributes_written_and_absent(self):
        # event.2, event.3: each key is written by its rules or absent.
        _, attributes = event()
        for key, _, rule in spec()["Attributes"]["records"]:
            if "{member}" in key:
                continue
            with self.subTest(key=key):
                if rule == "event.3":
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
            write(directory, lines(ENTRY, second), documents)
            found = raw_api_bodies.logs_data(directory)
        ids = []
        for data in found:
            (log_record,) = data["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
            for attribute in log_record["attributes"]:
                if attribute["key"] == "gen_ai.conversation.id":
                    ids.append(attribute["value"]["stringValue"])
        self.assertEqual(ids, ["s1", "s2"])

    def test_2_session_id_and_the_rest(self):
        _, attributes = event()
        self.assertEqual(attributes["gen_ai.conversation.id"], {"stringValue": "s1"})
        self.assertNotIn("anthropic.claude_code.session_id", attributes)
        for member in ENTRY:
            if member != "session_id":
                with self.subTest(member):
                    key = "anthropic.claude_code." + member
                    self.assertEqual(attributes[key], {"stringValue": ENTRY[member]})

    def test_4_rejected_naming_the_line(self):
        cases = [
            ("[1]\n", {}),
            (lines(dict(ENTRY, request_file=1)), {}),
            (lines(ENTRY), {"a.request.json": "[]"}),
            (lines(ENTRY), {"a.request.json": dict(REQUEST, max_tokens=1.5)}),
            (lines(ENTRY) + "\n", {}),
        ]
        for index, documents in cases:
            with self.subTest(index=index, documents=list(documents)):
                for error in rejected(self, index, documents):
                    expected = 2 if index.endswith("\n\n") else 1
                    self.assertEqual(error.lineno, expected)
                    self.assertIsInstance(error, ValueError)

    def test_4_what_the_core_rejects_names_the_line(self):
        # A function tool's name is a string, which the core validates.
        second = dict(ENTRY, request_file="b.request.json")
        documents = {
            "a.request.json": REQUEST,
            "b.request.json": dict(REQUEST, tools=[{"type": "custom", "name": 1}]),
            "req_1.response.json": RESPONSE,
        }
        with tempfile.TemporaryDirectory() as directory:
            write(directory, lines(ENTRY, second), documents)
            with self.assertRaises(raw_api_bodies.RawAPIBodiesDecodeError) as caught:
                raw_api_bodies.load(directory)
        self.assertEqual(caught.exception.lineno, 2)

    def test_4_a_file_not_held_is_absent_and_reported(self):
        # As in Claude Code's own index: a line naming no request file,
        # and a line whose response file is not yet on disk.
        no_request = {k: v for k, v in ENTRY.items() if k != "request_file"}
        not_yet = dict(ENTRY, response_file="later.response.json")
        documents = {"a.request.json": REQUEST, "req_1.response.json": RESPONSE}
        with tempfile.TemporaryDirectory() as directory:
            write(directory, lines(no_request, not_yet), documents)
            with self.assertLogs("living_memory.integrations", "WARNING") as logs:
                first, second = raw_api_bodies.logs_data(directory)
        self.assertEqual(
            logs.output,
            [
                "WARNING:living_memory.integrations:line 1: request_file not held",
                "WARNING:living_memory.integrations:line 2: response_file not held",
            ],
        )
        keys = [
            {a["key"] for a in data["resourceLogs"][0]["scopeLogs"][0]
             ["logRecords"][0]["attributes"]}
            for data in (first, second)
        ]
        self.assertNotIn("gen_ai.input.messages", keys[0])
        self.assertIn("gen_ai.output.messages", keys[0])
        self.assertIn("gen_ai.input.messages", keys[1])
        self.assertNotIn("gen_ai.output.messages", keys[1])

    def test_5_of_members_sharing_a_name_the_last(self):
        text = json.dumps(REQUEST)[:-1] + ', "model": "last"}'
        _, attributes = event(request=text)
        self.assertEqual(attributes["gen_ai.request.model"], {"stringValue": "last"})


class TestEvent(unittest.TestCase):
    def test_1_one_event_and_nothing_else(self):
        log_record, attributes = event()
        self.assertEqual(set(log_record), {"eventName", "attributes"})
        self.assertEqual(
            log_record["eventName"], "gen_ai.client.inference.operation.details"
        )
        self.assertIn("gen_ai.provider.name", attributes)


class TestLoad(unittest.TestCase):
    def test_the_core_reads_what_the_reader_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            documents = {"a.request.json": REQUEST, "req_1.response.json": RESPONSE}
            write(directory, lines(ENTRY), documents)
            sheets = {s["sheet name"]: s for s in raw_api_bodies.load(directory)}
        messages = sheets["gen_ai.input.messages"]["records"]
        self.assertEqual([row[2] for row in messages], ["user", "assistant", "user"])
        calls = sheets["gen_ai.input.messages.parts.tool_call"]["records"]
        self.assertEqual(calls[0][2:], ["t1", "get"])

    def test_an_empty_index_is_no_event(self):
        with tempfile.TemporaryDirectory() as directory:
            write(directory, "", {})
            self.assertEqual(raw_api_bodies.logs_data(directory), [])


if __name__ == "__main__":
    unittest.main()
