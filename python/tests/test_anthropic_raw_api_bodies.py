"""Tests of integrations.anthropic.claude_code.raw_api_bodies."""

import tempfile
import unittest
from pathlib import Path

from living_memory import _json, _module
from living_memory.integrations.anthropic.claude_code import raw_api_bodies as module

INDEX = [
    b'{"session_id":"s","query_source":"q","request_file":"1.request.json",'
    b'"response_file":"1.response.json"}',
    b'{"session_id":"s","query_source":"q","response_file":"x.response.json"}',
    b'{"session_id":"s","query_source":"q","request_file":"2.request.json",'
    b'"response_file":"2.response.json"}',
]
FILES = {
    "1.request.json": b'{"system":[{"type":"text","text":"be brief"}],"tools":[],'
    b'"messages":[{"role":"user","content":"hi"}]}',
    "1.response.json": b'{"content":[{"type":"text","text":"hello"}]}',
    "x.response.json": b'{"content":[]}',
    "2.request.json": b'{"system":[{"type":"text","text":"be brief"}],"tools":[],'
    b'"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":'
    b'[{"type":"text","text":"hello"}]},{"role":"user","content":"again"}]}',
    "2.response.json": b'{"content":[{"type":"text","text":"sure"}]}',
}


def recording(folder: Path, lines: list[bytes]) -> Path:
    (folder / module.FILE).write_bytes(b"\n".join(lines) + b"\n")
    for name, data in FILES.items():
        (folder / name).write_bytes(data)
    return folder


def given(path: Path, held=None) -> list[tuple]:
    return [
        (v["request"]["pointer"], v["version"], v["from"], v["line"])
        for v in map(_json.decode, module.values(path, held))
    ]


class TestValues(unittest.TestCase):
    # values.1-3, recording.4: each unit once, in order; a line with no
    # request file gives none; the re-sent message and the response it
    # was are one unit.
    def test_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            self.assertEqual(
                given(path),
                [
                    ("/system/0", "0", "1", "1"),
                    ("/messages/0", "0", "1", "1"),
                    ("/messages/1", "0", "2", "1"),
                    ("/messages/2", "0", "3", "3"),
                    ("/messages/3", "0", "4", "3"),
                ],
            )

    # values.3: a unit changed at its pointer is its next version.
    def test_version(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            changed = FILES["2.request.json"].replace(b'"hi"', b'"hi!"')
            (path / "2.request.json").write_bytes(changed)
            self.assertIn(("/messages/0", "1", "3", "3"), given(path))

    # values.4: only the lines after the last one held; the first new
    # line compared with its conversation's last held request.
    def test_only_new(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            held = [
                {
                    "session_id": "s",
                    "query_source": "q",
                    "request.pointer": p,
                    "version": "0",
                    "line": "1",
                }
                for p in ("/system/0", "/messages/0", "/messages/1")
            ]
            self.assertEqual(
                given(path, held),
                [("/messages/2", "0", "3", "3"), ("/messages/3", "0", "4", "3")],
            )

    # recording.4: a line whose files are not yet written stops the
    # read.
    def test_files_not_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = recording(Path(folder), INDEX)
            (path / "2.response.json").unlink()
            self.assertEqual(len(given(path)), 3)


class TestSchema(unittest.TestCase):
    # values.2: the spine's names are required, and the core reads it.
    def test_schema(self):
        root = _module.read(module.schema())
        self.assertEqual(root["title"], "raw_api_bodies")
        self.assertEqual(
            root["required"],
            ["session_id", "query_source", "request", "version", "from", "line"],
        )


if __name__ == "__main__":
    unittest.main()
