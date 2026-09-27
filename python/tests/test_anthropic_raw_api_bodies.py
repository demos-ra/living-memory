"""Tests of integrations.anthropic.claude_code.raw_api_bodies."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _json, _json_pointer, _schema
from living_memory.integrations.anthropic import messages
from living_memory.integrations.anthropic.claude_code import raw_api_bodies as module

NAMES = b'"model":"m","timestamp":"2026-09-27T10:00:00.000Z",'
INPUT = "2026-09-27/s"
INDEX = [
    b'{"session_id":"s","query_source":"q",'
    + NAMES
    + b'"request_file":"1.request.json",'
    b'"response_file":"1.response.json"}',
    b'{"session_id":"s","query_source":"q",'
    + NAMES
    + b'"response_file":"x.response.json"}',
    b'{"session_id":"s","query_source":"q",'
    + NAMES
    + b'"request_file":"2.request.json",'
    b'"response_file":"2.response.json"}',
    b'{"session_id":"s","query_source":"p",'
    + NAMES
    + b'"request_file":"3.request.json",'
    b'"response_file":"3.response.json"}',
]
SYSTEM = b'"system":[{"type":"text","text":"be brief"}],"tools":[],'
FILES = {
    "1.request.json": b"{" + SYSTEM + b'"messages":[{"role":"user","content":'
    b'[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}',
    "1.response.json": b'{"content":[{"type":"text","text":"hello"}],'
    b'"stop_reason":"end_turn","stop_sequence":null}',
    "x.response.json": b'{"content":[]}',
    "2.request.json": b"{" + SYSTEM + b'"messages":[{"role":"user","content":'
    b'[{"type":"text","text":"hi"}]},{"role":"assistant","content":'
    b'[{"type":"text","text":"hello"}]},{"role":"user","content":"again"}]}',
    "2.response.json": b'{"content":[{"type":"text","text":"sure"}]}',
    "3.request.json": b"{" + SYSTEM + b'"messages":[{"role":"user","content":'
    b'[{"type":"text","text":"hi"}]},{"role":"assistant","content":'
    b'[{"type":"text","text":"hello"}]},{"role":"user","content":"suggest"}]}',
    "3.response.json": b'{"content":[{"type":"text","text":"try this"}]}',
}


def recording(folder: Path, lines: list[bytes]) -> Path:
    (folder / module.FILE).write_bytes(b"\n".join(lines) + b"\n")
    for name, data in FILES.items():
        (folder / name).write_bytes(data)
    return folder


def given(path: Path, held=None) -> list[dict]:
    return [_json.decode(v) for v in module.values(path, INPUT, held)]


def rebuilt(found: list[dict]) -> dict[str, tuple]:
    # Each request, and its response's content, rebuilt from the values
    # alone: the first messages of the request it extends then its own,
    # that request's system blocks and tools but where it gives its own.
    requests: dict[str, dict] = {}
    for value in found:
        parent = requests.get(
            value["extends"], {"messages": [], "system": [], "tools": []}
        )
        whole = parent["messages"] + [parent.get("answer")]
        request = {
            "messages": whole[: int(value["kept"])],
            **{n: parent[n][: int(value["count"][n])] for n in ("system", "tools")},
        }
        for unit in value["units"]:
            member = next(k for k in unit if k != "request")
            tokens = _json_pointer.tokens(unit["request"]["pointer"])
            if member == "content":
                request["answer"] = {"role": "assistant", "content": unit[member]}
            elif int(tokens[1]) < len(request[member]):
                request[member][int(tokens[1])] = unit[member]
            else:
                request[member].append(unit[member])
        requests[value["index_line"]] = request
    return {
        line: (r["system"], r["tools"], r["messages"], r["answer"])
        for line, r in requests.items()
    }


class TestValues(unittest.TestCase):
    # values.1, values.2, recording.5: one value per request, in order;
    # a line with no request file gives none; a request keeps its
    # parent's messages and its response, and gives only what follows.
    def test_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            found = given(recording(Path(folder), INDEX[:3]))
            self.assertEqual(
                [(v["index_line"], v["extends"], v["kept"]) for v in found],
                [("1", "0", "0"), ("3", "1", "2")],
            )
            pointers = [u["request"]["pointer"] for u in found[1]["units"]]
            self.assertEqual(pointers, ["/messages/2", "/messages/3"])

    # values.2: a side thread's request extends the main thread's, its
    # parent in another thread of the session.
    def test_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            found = given(recording(Path(folder), INDEX))
            self.assertEqual(
                (found[2]["query_source"], found[2]["extends"], found[2]["kept"]),
                ("p", "3", "2"),
            )

    # values.3, messages.mtsv kept.2: a cache control breakpoint is
    # left out, and does not make a message a new one.
    def test_cache_control(self):
        with tempfile.TemporaryDirectory() as folder:
            found = given(recording(Path(folder), INDEX[:3]))
            first = next(u for u in found[0]["units"] if "messages" in u)
            self.assertNotIn("cache_control", first["messages"]["content"][0])
            self.assertEqual(found[1]["kept"], "2")

    # values.3: a message is compared whole, so one that differs from
    # the parent's only in a member beside its role and content is its
    # own again.
    def test_whole_message(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX[:3])
            changed = FILES["2.request.json"].replace(
                b'"content":"again"', b'"content":"again","output_config":{}'
            )
            first = changed.replace(
                b'"text":"hi"}]}', b'"text":"hi"}],"output_config":{}}', 1
            )
            (path / "2.request.json").write_bytes(first)
            self.assertEqual(given(path)[1]["kept"], "0")

    # values.4: the request's names and counts, and the response's
    # members kept beside its content, null ones left out.
    def test_names(self):
        with tempfile.TemporaryDirectory() as folder:
            first = given(recording(Path(folder), INDEX[:1]))[0]
            self.assertEqual(
                (first["model"], first["timestamp"]), ("m", "2026-09-27T10:00:00.000Z")
            )
            self.assertEqual(first["count"], {"system": "1", "tools": "0"})
            self.assertEqual(first["stop_reason"], "end_turn")
            self.assertNotIn("stop_sequence", first)

    # values.5: every request and its response rebuild exactly from the
    # values, each unit as it is kept.
    def test_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            found = rebuilt(given(path))
            for line, name in (("1", "1"), ("3", "2"), ("4", "3")):
                request = _json.decode(FILES[f"{name}.request.json"])
                response = _json.decode(FILES[f"{name}.response.json"])
                expected = (
                    [messages.kept("system", b) for b in request["system"]],
                    [],
                    [messages.kept("messages", m) for m in request["messages"]],
                    {"role": "assistant", "content": response["content"]},
                )
                self.assertEqual(found[line], expected)

    # values.6: only the lines after the last one held, the parent found
    # among the latest requests held, read again from their files.
    def test_only_new(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX[:3])
            found = given(path, [{"index_line": "1"}])
            self.assertEqual(
                [(v["index_line"], v["extends"], v["kept"]) for v in found],
                [("3", "1", "2")],
            )

    # recording.4: a line whose files are not yet written stops the
    # read.
    def test_files_not_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            (path / "2.response.json").unlink()
            self.assertEqual(len(given(path)), 1)


class TestInputs(unittest.TestCase):
    # values.7: one input per session, named by the date of its first
    # line and its session_id, in the order of first lines; each
    # session's values apart.
    def test_sessions(self):
        with tempfile.TemporaryDirectory() as folder:
            other = INDEX[0].replace(b'"s"', b'"t"').replace(b"09-27", b"09-28")
            path = recording(Path(folder), [INDEX[0], other, *INDEX[1:3]])
            names = module.inputs(path)
            self.assertEqual(names, ["2026-09-27/s", "2026-09-28/t"])
            self.assertEqual(module.input_of(names, "t"), "2026-09-28/t")
            lines = [v["index_line"] for v in given(path)]
            self.assertEqual(lines, ["1", "4"])


class TestSpent(unittest.TestCase):
    # recording.6: the files of every line held are spent, but the
    # latest request of each thread held; index.jsonl never.
    def test_spent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX[:3])
            spent = module.spent(path, INPUT, [{"index_line": "3"}])
            names = sorted(p.name for p in spent)
            self.assertEqual(
                names, ["1.request.json", "1.response.json", "x.response.json"]
            )


class TestMap(unittest.TestCase):
    RECORDS = [
        {
            "index_line": "1",
            "session_id": "s",
            "query_source": "q",
            "timestamp": "a",
            "extends": "0",
        },
        {
            "index_line": "3",
            "session_id": "s",
            "query_source": "q",
            "timestamp": "b",
            "extends": "1",
        },
        {
            "index_line": "4",
            "session_id": "s",
            "query_source": "p",
            "timestamp": "c",
            "extends": "3",
        },
        {
            "index_line": "5",
            "session_id": "s",
            "query_source": "q",
            "timestamp": "d",
            "extends": "0",
        },
    ]

    # map.1: grouped by session_id and query_source, in the order of
    # their first lines.
    def test_groups(self):
        self.assertEqual(
            module.groups(self.RECORDS[:3]),
            [["s", "q", "2", "1", "3", "a", "b"], ["s", "p", "1", "4", "4", "c", "c"]],
        )

    # map.2: the lineage of the latest request, earliest first, ending
    # at a request that extends none.
    def test_lineage(self):
        self.assertEqual(
            [r["index_line"] for r in module.lineage(self.RECORDS[:3])], ["1", "3", "4"]
        )
        self.assertEqual([r["index_line"] for r in module.lineage(self.RECORDS)], ["5"])


class TestSchema(unittest.TestCase):
    # values.4: a value's names are required, and the core reads it.
    def test_schema(self):
        root = _schema.read(module.schema())
        self.assertEqual(root["title"], "raw_api_bodies")
        expected = ["index_line", "session_id", "query_source", "model", "timestamp"]
        expected += ["extends", "kept", "count", "units"]
        self.assertEqual(root["required"], expected)


if __name__ == "__main__":
    unittest.main()
