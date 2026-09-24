"""Tests of providers/anthropic/claude_code/install, rule by rule."""

import json
import unittest
from importlib.resources import files
from pathlib import Path

import mtsv
from living_memory.integrations.providers.anthropic.claude_code import install

from support import ROOT

HOME = Path("/home/u")
BODIES = Path("anthropic", "claude_code", "raw_api_bodies")
DATA = HOME / ".local" / "share" / "living-memory"
PLUGIN = ROOT / "plugins" / "claude-code"


def spec():
    with files(install.__package__).joinpath("install.mtsv").open("rb") as f:
        return {sheet["sheet name"]: sheet for sheet in mtsv.load(f)}


class TestSpecification(unittest.TestCase):
    def test_the_rules_in_the_order_of_the_conformance_clause(self):
        ids = [row[0] for row in spec()["Rules"]["records"]]
        self.assertEqual(
            ids,
            [
                "prompt.1",
                "directory.1",
                "directory.2",
                "plugin.1",
                "capture.1",
                "hook.1",
                "hook.2",
                "hook.3",
            ],
        )


class TestDirectory(unittest.TestCase):
    def test_1_linux(self):
        cases = [
            ({}, DATA),
            ({"XDG_DATA_HOME": ""}, DATA),
            ({"XDG_DATA_HOME": "relative"}, DATA),
            ({"XDG_DATA_HOME": "/d"}, Path("/d/living-memory")),
        ]
        for environ, expected in cases:
            with self.subTest(environ):
                self.assertEqual(
                    install.data_directory(environ, "linux", HOME), expected
                )

    def test_1_macos(self):
        self.assertEqual(
            install.data_directory({}, "darwin", HOME),
            HOME / "Library" / "Application Support" / "living-memory",
        )

    def test_1_any_other_system_refused(self):
        with self.assertRaises(LookupError):
            install.data_directory({}, "win32", HOME)

    def test_2_directories(self):
        self.assertEqual(
            install.directories(DATA),
            [
                DATA,
                DATA / "anthropic",
                DATA / "anthropic" / "claude_code",
                DATA / BODIES,
            ],
        )

    def test_2_index_file(self):
        self.assertEqual(
            install.index_file(DATA), DATA / BODIES / "index.jsonl"
        )


class TestPrompt(unittest.TestCase):
    def test_1_each_change_stated(self):
        found = install.changes(DATA, HOME)
        self.assertIn(str(DATA / BODIES), found)
        self.assertIn(str(DATA / BODIES) + ".mtsv", found)
        for command in install.commands(DATA):
            self.assertIn(" ".join(command), found)
        self.assertIn("OTEL_LOG_RAW_API_BODIES", found)
        self.assertIn(str(HOME / ".claude" / "settings.json"), found)


class TestPlugin(unittest.TestCase):
    def test_1_marketplace_then_plugin(self):
        self.assertEqual(
            install.commands(DATA),
            [
                ["claude", "plugin", "marketplace", "add", "demos-ra/living-memory"],
                [
                    "claude",
                    "plugin",
                    "install",
                    "claude-code@living-memory",
                    "--config",
                    f"data_dir={DATA}",
                ],
            ],
        )


class TestCapture(unittest.TestCase):
    def test_1_set_and_the_rest_kept(self):
        document = '{"model": "x", "env": {"A": "1"}}'
        found = json.loads(install.settings(document, DATA))
        self.assertEqual(found["model"], "x")
        self.assertEqual(
            found["env"],
            {"A": "1", "OTEL_LOG_RAW_API_BODIES": f"file:{DATA / BODIES}"},
        )

    def test_1_a_file_that_does_not_exist(self):
        found = json.loads(install.settings(None, DATA))
        self.assertEqual(list(found), ["env"])

    def test_1_not_an_object(self):
        for document in ("[]", '{"env": []}', "x", '{"a": NaN}'):
            with self.subTest(document):
                with self.assertRaises(ValueError):
                    install.settings(document, DATA)


class TestHook(unittest.TestCase):
    def test_1_the_option_and_no_version(self):
        manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        self.assertEqual(manifest["name"], "claude-code")
        self.assertNotIn("version", manifest)
        option = manifest["userConfig"]["data_dir"]
        self.assertEqual((option["type"], option["required"]), ("directory", True))

    def test_2_stop(self):
        (hook,) = hooks("Stop")
        self.assertEqual(
            hook,
            {
                "type": "command",
                "command": "living-memory",
                "args": [
                    "${user_config.data_dir}/anthropic/claude_code/raw_api_bodies"
                ],
                "async": True,
            },
        )

    def test_3_session_start(self):
        (hook,) = hooks("SessionStart")
        self.assertEqual(hook["command"], "echo")
        self.assertEqual(
            hook["args"],
            [
                "Claude Code's past conversations are kept as MTSV in"
                " ${user_config.data_dir}/anthropic/claude_code/"
                "raw_api_bodies.mtsv, written after each response."
            ],
        )

    def test_the_catalog_lists_the_plugin(self):
        catalog = json.loads(
            (ROOT / ".claude-plugin" / "marketplace.json").read_text()
        )
        self.assertEqual(catalog["name"], "living-memory")
        (entry,) = catalog["plugins"]
        self.assertEqual(
            (entry["name"], entry["source"]), ("claude-code", "./plugins/claude-code")
        )


def hooks(event):
    found = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    (matcher,) = found[event]
    return matcher["hooks"]


if __name__ == "__main__":
    unittest.main()
