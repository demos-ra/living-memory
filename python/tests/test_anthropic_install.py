"""Tests of integrations.anthropic.claude_code.install."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mtsv

from living_memory import _json
from living_memory.integrations.anthropic.claude_code import install as module

SETTINGS = (
    b'{\n  "hooks": {\n    "Stop": [\n      {\n        "hooks": []\n'
    b"      }\n    ]\n  }\n}\n"
)


def installed(home: str, options: list[str]) -> tuple[bytes, dict]:
    # The user settings after two installs, the second adding nothing.
    settings = Path(home, ".claude", "settings.json")
    settings.parent.mkdir(exist_ok=True)
    settings.write_bytes(SETTINGS)
    environ = {"XDG_DATA_HOME": f"{home}/data", "CLAUDE_CONFIG_DIR": ""}
    with mock.patch.object(Path, "home", return_value=Path(home)):
        with mock.patch.dict(os.environ, environ):
            with mock.patch.object(module.sys, "platform", "linux"):
                module.install(options)
                once = settings.read_bytes()
                module.install(options)
                assert settings.read_bytes() == once
    return once, _json.decode(once)


class TestInstall(unittest.TestCase):
    # install.3, hooks.1-3: the capture variable and three matcher groups
    # are added, what the settings held is kept, and a second install
    # adds nothing.
    def test_adds_once(self):
        with tempfile.TemporaryDirectory() as home:
            _, written = installed(home, [])
            folder = f"{home}/data/living-memory/anthropic/claude_code/raw_api_bodies"
            self.assertEqual(
                written["env"]["OTEL_LOG_RAW_API_BODIES"], f"file:{folder}"
            )
            self.assertEqual(written["hooks"]["Stop"][0], {"hooks": []})
            self.assertEqual(written["hooks"]["Stop"][1]["hooks"][0]["args"], [folder])
            context = ["--add-context=claude-code", folder]
            for event in ("SessionStart", "UserPromptSubmit"):
                with self.subTest(event=event):
                    self.assertEqual(
                        written["hooks"][event][0]["hooks"][0]["args"], context
                    )
            self.assertEqual(os.stat(folder).st_mode & 0o777, 0o700)

    # install.5: kept, each hook runs living-memory with --keep-files.
    def test_keep_files(self):
        with tempfile.TemporaryDirectory() as home:
            _, written = installed(home, ["--keep-files"])
            for event in ("Stop", "SessionStart", "UserPromptSubmit"):
                with self.subTest(event=event):
                    args = written["hooks"][event][-1]["hooks"][0]["args"]
                    self.assertEqual(args[0], "--keep-files")

    # capture.2: the user settings are in CLAUDE_CONFIG_DIR where it is
    # set.
    def test_config_dir(self):
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": "/config"}):
            self.assertEqual(module._settings(), Path("/config/settings.json"))

    # folder.1: a relative XDG_DATA_HOME is ignored.
    def test_relative_data_home(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}):
            with mock.patch.object(module.sys, "platform", "linux"):
                self.assertEqual(module._data_home(), Path.home() / ".local" / "share")

    # install.2, capture.3, install.5: the change is stated, the consent,
    # the start at the next session and the files removed included; the
    # question asks whether to keep them.
    def test_change(self):
        with mock.patch.object(module.sys, "platform", "linux"):
            stated = module.change()
        for part in ("tool content", "from the next session", "unless you keep them"):
            with self.subTest(part=part):
                self.assertIn(part, stated)
        self.assertIn("raw API bodies", module.question())

    # folder.2: on Windows the change is refused, and nothing is made.
    def test_refused_on_windows(self):
        with mock.patch.object(module.sys, "platform", "win32"):
            for step in (module.change, lambda: module.install([])):
                with self.assertRaises(OSError):
                    step()


HEADER = ["pointer", "index_line", "session_id", "query_source", "timestamp", "extends"]
HELD = [
    dict(zip(HEADER, ["/0", "1", "s", "q", "2026-09-27T01", "0"])),
    dict(zip(HEADER, ["/1", "2", "s", "q", "2026-09-27T02", "1"])),
]
SHEETS = [
    {
        "file": "0 raw_api_bodies.mtsv",
        "name": "raw_api_bodies",
        "header": HEADER,
        "positions": [0, 1],
    },
    {
        "file": "1 BetaMessageParam.mtsv",
        "name": "BetaMessageParam",
        "header": ["parent", "pointer", "role"],
        "positions": [0, 0, 1],
    },
]


def look(given: int):
    def outputs(name):
        return {"sheets": lambda: SHEETS, "held": lambda: HELD, "given": lambda: given}

    return outputs


def hook(event: str, source: str = "startup") -> bytes:
    return _json.encode({"session_id": "s", "hook_event_name": event, "source": source})


class TestContext(unittest.TestCase):
    NAMES = ["2026-09-27/s"]

    # context.1, context.4: from SessionStart, the store, its dates, the
    # date's sessions grouped, and the session's sheet files and columns;
    # every request given.
    def test_start(self):
        text, marks = module.context(
            hook("SessionStart"), Path("/out"), self.NAMES, look(0)
        )
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(
            list(sheets),
            [
                "store",
                "raw_api_bodies by date",
                "raw_api_bodies by session_id, query_source",
                "store by sheet",
                "store columns",
            ],
        )
        self.assertEqual(
            sheets["raw_api_bodies by date"]["records"], [["2026-09-27", "1", "2"]]
        )
        self.assertEqual(sheets["store by sheet"]["records"][1][2], "3")
        self.assertEqual(marks, {"2026-09-27/s": 2})

    # context.2: after compaction, the lineage of the latest request.
    def test_compact(self):
        text, _ = module.context(
            hook("SessionStart", "compact"), Path("/out"), self.NAMES, look(0)
        )
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(
            sheets["raw_api_bodies not kept"]["records"],
            [["s", "q", "2", "1", "2", "2026-09-27T01", "2026-09-27T02"]],
        )

    # context.3: from UserPromptSubmit, only the requests not yet given,
    # and where their records are, counted from 1; with none, nothing.
    def test_turn(self):
        text, marks = module.context(
            hook("UserPromptSubmit"), Path("/out"), self.NAMES, look(1)
        )
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(sheets["raw_api_bodies"]["records"], [list(HELD[1].values())])
        self.assertEqual(
            sheets["store by sheet"]["records"],
            [
                ["0 raw_api_bodies.mtsv", "raw_api_bodies", "1", "2", "2"],
                ["1 BetaMessageParam.mtsv", "BetaMessageParam", "1", "3", "3"],
            ],
        )
        self.assertEqual(marks, {"2026-09-27/s": 2})
        nothing, _ = module.context(
            hook("UserPromptSubmit"), Path("/out"), self.NAMES, look(2)
        )
        self.assertEqual(nothing, "")


if __name__ == "__main__":
    unittest.main()
