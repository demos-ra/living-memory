"""Tests of integrations.anthropic.claude_code.install."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mtsv

from living_memory import _communication, _json
from living_memory.integrations.anthropic.claude_code import install as module

FROM = "--from=anthropic/claude_code/raw_api_bodies"
SETTINGS = (
    b'{\n  "hooks": {\n    "Stop": [\n      {\n        "hooks": []\n'
    b"      }\n    ]\n  }\n}\n"
)


def installed(home: str, options: list[str], fresh: bool = True) -> tuple[bytes, dict]:
    # The user settings after two installs, the second adding nothing;
    # fresh, the settings are first those of SETTINGS.
    settings = Path(home, ".claude", "settings.json")
    settings.parent.mkdir(exist_ok=True)
    if fresh:
        settings.write_bytes(SETTINGS)
    environ = {
        "XDG_DATA_HOME": f"{home}/data",
        "XDG_CONFIG_HOME": "",
        "CLAUDE_CONFIG_DIR": "",
    }
    with mock.patch.object(Path, "home", return_value=Path(home)):
        with mock.patch.dict(os.environ, environ):
            with mock.patch.object(module.sys, "platform", "linux"):
                module.install(options)
                once = settings.read_bytes()
                module.install(options)
                assert settings.read_bytes() == once
    return once, _json.decode(once)


class TestInstall(unittest.TestCase):
    # install.3, hooks.1-3, hooks.5, folder.4: the capture variable and
    # three matcher groups are added, each naming its reader and the
    # data bank, what the settings held is kept, and a second install
    # adds nothing; each folder made is readable by the user alone.
    def test_adds_once(self):
        with tempfile.TemporaryDirectory() as home:
            _, written = installed(home, [])
            folder = f"{home}/data/living-memory/anthropic/claude_code/raw_api_bodies"
            bank = f"{home}/Documents/living-memory/anthropic/claude_code"
            output = f"--output={bank}/raw_api_bodies.mtsv"
            self.assertEqual(
                written["env"]["OTEL_LOG_RAW_API_BODIES"], f"file:{folder}"
            )
            self.assertEqual(written["hooks"]["Stop"][0], {"hooks": []})
            self.assertEqual(
                written["hooks"]["Stop"][1]["hooks"][0]["args"], [FROM, output, folder]
            )
            context = [FROM, "--add-context=claude-code", output, folder]
            for event in ("SessionStart", "UserPromptSubmit"):
                with self.subTest(event=event):
                    self.assertEqual(
                        written["hooks"][event][0]["hooks"][0]["args"], context
                    )
            made = [folder, f"{home}/data/living-memory", f"{bank}/raw_api_bodies.mtsv"]
            made += [f"{home}/Documents/living-memory", f"{home}/Documents"]
            for one in made:
                with self.subTest(folder=one):
                    self.assertEqual(os.stat(one).st_mode & 0o777, 0o700)

    # install.5: kept, each hook runs living-memory with --keep-files.
    def test_keep_files(self):
        with tempfile.TemporaryDirectory() as home:
            _, written = installed(home, ["--keep-files"])
            for event in ("Stop", "SessionStart", "UserPromptSubmit"):
                with self.subTest(event=event):
                    args = written["hooks"][event][-1]["hooks"][0]["args"]
                    self.assertEqual(args[0], "--keep-files")

    # install.3: a group install added before is replaced where it
    # stands, not added beside, and another group is kept.
    def test_replaces_its_own(self):
        with tempfile.TemporaryDirectory() as home:
            installed(home, ["--keep-files"])
            _, written = installed(home, [], fresh=False)
            stop = written["hooks"]["Stop"]
            self.assertEqual(len(stop), 2)
            self.assertEqual(stop[0], {"hooks": []})
            self.assertNotIn("--keep-files", stop[1]["hooks"][0]["args"])
            self.assertEqual(len(written["hooks"]["UserPromptSubmit"]), 1)

    # capture.2: the user settings are in CLAUDE_CONFIG_DIR where it is
    # set.
    def test_config_dir(self):
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": "/config"}):
            self.assertEqual(module._settings(), Path("/config/settings.json"))

    # folder.4: the user's documents directory is ~/Documents, on Linux
    # as the last line of user-dirs.dirs that names it gives it, under
    # the configuration home.
    def test_documents(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.object(Path, "home", return_value=Path(home)):
                with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}):
                    with mock.patch.object(module.sys, "platform", "linux"):
                        self.assertEqual(module._documents(), Path(home, "Documents"))
                        Path(home, ".config").mkdir()
                        Path(home, ".config", "user-dirs.dirs").write_text(
                            'XDG_DOCUMENTS_DIR="$HOME/Dokumente"\n'
                            'XDG_DOCUMENTS_DIR="relative"\n'
                            "# a comment\n"
                        )
                        self.assertEqual(module._documents(), Path(home, "Dokumente"))
                    with mock.patch.object(module.sys, "platform", "darwin"):
                        self.assertEqual(module._documents(), Path(home, "Documents"))

    # folder.1: a relative XDG_DATA_HOME is ignored.
    def test_relative_data_home(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}):
            with mock.patch.object(module.sys, "platform", "linux"):
                self.assertEqual(module._data_home(), Path.home() / ".local" / "share")

    # install.2, capture.3, install.5: the change is stated, the
    # consent, the start at the next session and the files removed
    # included; the question asks whether to keep them.
    def test_change(self):
        with mock.patch.object(module.sys, "platform", "linux"):
            stated = module.change()
        parts = ("tool content", "from the next session", "unless you keep them")
        parts += ("among your documents",)
        for part in parts:
            with self.subTest(part=part):
                self.assertIn(part, stated)
        self.assertIn("raw API bodies", module.question())

    # folder.2: on Windows the change is refused, and nothing is made.
    def test_refused_on_windows(self):
        with mock.patch.object(module.sys, "platform", "win32"):
            for step in (module.change, lambda: module.install([])):
                with self.assertRaises(OSError):
                    step()


HEADER = ["_input value", "index_line", "session_id", "query_source"]
HEADER += ["timestamp", "extends"]


def value(i: int, extends: int) -> list[str]:
    return [str(i), str(i + 1), "s", "q", f"2026-09-27T{i:02}", str(extends)]


class Bank:
    # A data bank of one input, whose stored sheets are given, answering
    # only by the core's communications (living-memory.mtsv ›
    # communication.3-6).
    def __init__(self, records: list[list[str]]) -> None:
        roots = {"sheet name": "raw_api_bodies", "header": HEADER, "records": records}
        messages = {
            "sheet name": "BetaMessageParam",
            "header": ["_input value", "_instance", "_parent", "_pointer", "role"],
            "records": [[r[0], "2", "0", "/units/0", "user"] for r in records],
        }
        self.stored = [(0, roots), (154, messages)]
        self.values = len(records)
        self.communicated = 0

    def names(self) -> str:
        return _communication.names([("2026-09-27/s", self.stored, self.values)])

    def filter(self, name, values, places):
        first, last = values or (0, self.values - 1)
        return _communication.records(self.stored, first, last, places)

    def new(self, name, places):
        values = self.values
        text = _communication.records(
            self.stored, self.communicated, values - 1, places
        )
        self.communicated = values
        return text


def hook(event: str, source: str = "startup") -> bytes:
    return _json.encode({"session_id": "s", "hook_event_name": event, "source": source})


class TestContext(unittest.TestCase):
    # context.1, context.4: from SessionStart, the data bank, its dates,
    # the date's requests grouped, and the session's sheets and fields,
    # from the names; what is new then begins after the last value.
    def test_start(self):
        bank = Bank([value(0, 0), value(1, 1)])
        text = module.context(hook("SessionStart"), bank)
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(
            list(sheets),
            [
                "data bank",
                "raw_api_bodies by date",
                "raw_api_bodies by session_id, query_source",
                "_sheets",
                "_fields",
            ],
        )
        self.assertEqual(
            sheets["data bank"]["records"][0][1], "anthropic/claude_code/raw_api_bodies"
        )
        self.assertEqual(
            sheets["raw_api_bodies by date"]["records"], [["2026-09-27", "1", "2"]]
        )
        self.assertEqual(
            sheets["_sheets"]["records"],
            [
                ["2026-09-27/s", "0", "raw_api_bodies"],
                ["2026-09-27/s", "154", "BetaMessageParam"],
            ],
        )
        self.assertEqual(bank.communicated, 2)

    # context.2, context.7: after compaction, the lineage of the latest
    # request, grouped.
    def test_compact(self):
        bank = Bank([value(0, 0), value(1, 1)])
        text = module.context(hook("SessionStart", "compact"), bank)
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(
            sheets["raw_api_bodies not kept"]["records"],
            [["s", "q", "2", "1", "2", "2026-09-27T00", "2026-09-27T01"]],
        )

    # context.3: from UserPromptSubmit, what is new of the sheet of the
    # input values alone; with none, nothing.
    def test_turn(self):
        bank = Bank([value(0, 0), value(1, 1)])
        bank.communicated = 1
        text = module.context(hook("UserPromptSubmit"), bank)
        self.assertEqual(
            mtsv.loads(text),
            [
                {
                    "sheet name": "raw_api_bodies",
                    "header": HEADER,
                    "records": [value(1, 1)],
                }
            ],
        )
        self.assertEqual(module.context(hook("UserPromptSubmit"), bank), "")

    # context.5, context.6: requests that would pass the cap are given
    # as their groups, a role name qualifying each domain name.
    def test_cap(self):
        bank = Bank([value(i, 0) for i in range(400)])
        text = module.context(hook("UserPromptSubmit"), bank)
        sheets = {s["sheet name"]: s for s in mtsv.loads(text)}
        self.assertEqual(list(sheets), ["raw_api_bodies by session_id, query_source"])
        group = sheets["raw_api_bodies by session_id, query_source"]
        self.assertEqual(group["header"][3:5], ["first.index_line", "last.index_line"])
        self.assertEqual(group["records"][0][:5], ["s", "q", "400", "1", "400"])
        self.assertLessEqual(len(text), 10_000)


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

    # context.6: grouped by session_id and query_source, in the order of
    # their first lines.
    def test_groups(self):
        self.assertEqual(
            module.groups(self.RECORDS[:3]),
            [["s", "q", "2", "1", "3", "a", "b"], ["s", "p", "1", "4", "4", "c", "c"]],
        )

    # context.7: the lineage of the latest request, earliest first,
    # ending at a request that extends none.
    def test_lineage(self):
        self.assertEqual(
            [r["index_line"] for r in module.lineage(self.RECORDS[:3])], ["1", "3", "4"]
        )
        self.assertEqual([r["index_line"] for r in module.lineage(self.RECORDS)], ["5"])


if __name__ == "__main__":
    unittest.main()
