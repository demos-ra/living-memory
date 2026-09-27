"""Tests of integrations.anthropic.claude_code.install."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from living_memory import _json
from living_memory.integrations.anthropic.claude_code import install as module

SETTINGS = (
    b'{\n  "hooks": {\n    "Stop": [\n      {\n        "hooks": []\n'
    b"      }\n    ]\n  }\n}\n"
)


class TestInstall(unittest.TestCase):
    # install.3, hooks.1-3: the capture variable and two matcher groups
    # are added, what the settings held is kept, and a second install
    # adds nothing.
    def test_adds_once(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Path(home, ".claude", "settings.json")
            settings.parent.mkdir()
            settings.write_bytes(SETTINGS)
            environ = {"XDG_DATA_HOME": f"{home}/data"}
            with mock.patch.object(Path, "home", return_value=Path(home)):
                with mock.patch.dict(os.environ, environ):
                    with mock.patch.object(module.sys, "platform", "linux"):
                        module.install()
                        once = settings.read_bytes()
                        module.install()
                        self.assertEqual(settings.read_bytes(), once)
            written = _json.decode(once)
            folder = f"{home}/data/living-memory/anthropic/claude_code/raw_api_bodies"
            self.assertEqual(
                written["env"]["OTEL_LOG_RAW_API_BODIES"], f"file:{folder}"
            )
            self.assertEqual(written["hooks"]["Stop"][0], {"hooks": []})
            self.assertEqual(written["hooks"]["Stop"][1]["hooks"][0]["args"], [folder])
            self.assertEqual(len(written["hooks"]["SessionStart"]), 1)
            self.assertEqual(os.stat(folder).st_mode & 0o777, 0o700)

    # folder.1: a relative XDG_DATA_HOME is ignored.
    def test_relative_data_home(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}):
            with mock.patch.object(module.sys, "platform", "linux"):
                self.assertEqual(module._data_home(), Path.home() / ".local" / "share")

    # install.2: the change is stated, the consent included.
    def test_change(self):
        with mock.patch.object(module.sys, "platform", "linux"):
            self.assertIn(
                "your prompts, tool details and tool content", module.change()
            )

    # folder.2: on Windows the change is refused, and nothing is made.
    def test_refused_on_windows(self):
        with mock.patch.object(module.sys, "platform", "win32"):
            for step in (module.change, module.install):
                with self.subTest(step=step.__name__):
                    with self.assertRaises(OSError):
                        step()


if __name__ == "__main__":
    unittest.main()
