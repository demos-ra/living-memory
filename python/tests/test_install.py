"""Tests of providers/anthropic/claude_code/install, rule by rule."""

import json
import unittest
from importlib.resources import files
from pathlib import Path

import mtsv
from living_memory.providers.anthropic.claude_code import install

from support import ROOT

HOME = Path("/home/u")
BODIES = Path("anthropic", "claude_code", "raw_api_bodies")
DATA = HOME / ".local" / "share" / "living-memory"
SETTINGS = HOME / ".claude" / "settings.json"
PLUGIN = ROOT / "plugins" / "claude-code"


def spec():
    with files(install.__package__).joinpath("install.mtsv").open("rb") as f:
        return {sheet["sheet name"]: sheet for sheet in mtsv.load(f)}


def plan(environ=None, system="linux"):
    return install.plan(environ or {}, system, HOME)


def steps(action):
    return [step[1:] for step in plan()[1] if step[0] == action]


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


class TestOrder(unittest.TestCase):
    def test_directory_plugin_then_capture_last(self):
        actions = [step[0] for step in plan()[1]]
        self.assertEqual(
            actions,
            ["directory"] * 4 + ["file", "run", "run", "directory", "replace"],
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
                first = plan(environ)[1][0]
                self.assertEqual(first, ("directory", expected, 0o700))

    def test_1_macos(self):
        first = plan(system="darwin")[1][0]
        data = HOME / "Library" / "Application Support" / "living-memory"
        self.assertEqual(first, ("directory", data, 0o700))

    def test_1_any_other_system_refused(self):
        with self.assertRaises(LookupError):
            plan(system="win32")

    def test_2_directories_private_and_an_empty_index(self):
        self.assertEqual(
            steps("directory")[:4],
            [
                (DATA, 0o700),
                (DATA / "anthropic", 0o700),
                (DATA / "anthropic" / "claude_code", 0o700),
                (DATA / BODIES, 0o700),
            ],
        )
        self.assertEqual(steps("file"), [(DATA / BODIES / "index.jsonl", None)])


class TestPrompt(unittest.TestCase):
    def test_1_each_change_stated(self):
        found = plan()[0]
        self.assertIn(str(DATA / BODIES), found)
        self.assertIn(str(DATA / BODIES) + ".mtsv", found)
        for command, _ in steps("run"):
            self.assertIn(" ".join(command), found)
        self.assertIn("OTEL_LOG_RAW_API_BODIES", found)
        self.assertIn(str(SETTINGS), found)


class TestPlugin(unittest.TestCase):
    def test_1_marketplace_then_plugin(self):
        self.assertEqual(
            [command for command, _ in steps("run")],
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
    def rewrite(self, document):
        ((path, rewrite),) = steps("replace")
        self.assertEqual(path, SETTINGS)
        return rewrite(document)

    def test_1_set_and_the_rest_kept(self):
        document = '{"model": "x", "env": {"A": "1"}}'
        found = json.loads(self.rewrite(document))
        self.assertEqual(found["model"], "x")
        self.assertEqual(
            found["env"],
            {"A": "1", "OTEL_LOG_RAW_API_BODIES": f"file:{DATA / BODIES}"},
        )

    def test_1_a_file_that_does_not_exist(self):
        found = json.loads(self.rewrite(None))
        self.assertEqual(list(found), ["env"])

    def test_1_not_an_object(self):
        for document in ("[]", '{"env": []}', "x", '{"a": NaN}'):
            with self.subTest(document):
                with self.assertRaises(ValueError):
                    self.rewrite(document)


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
        catalog = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
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
