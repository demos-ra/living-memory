"""Tests of integrations.anthropic.messages: the kept units' schema."""

import unittest

from living_memory import _json, _json_schema, _schema
from living_memory.integrations.anthropic import messages as module


class TestUnits(unittest.TestCase):
    # kept.1: the four kept parts, each by its unit.
    def test_kept_parts(self):
        self.assertEqual(
            sorted(module.units()), ["content", "messages", "system", "tools"]
        )

    # kept.1: the response's members kept beside its content.
    def test_ending(self):
        self.assertEqual(
            list(module.ending()),
            [
                "stop_reason",
                "stop_sequence",
                "stop_details",
                "input_transformations",
                "context_management",
            ],
        )

    # kept.2: a cache control breakpoint is left out of every block,
    # those within a block's content and its source's content included,
    # and out of the schema; nothing else is touched.
    def test_cache_control(self):
        control = {"type": "ephemeral"}
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": control},
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "cache_control": control,
                    "content": [
                        {"type": "text", "text": "x", "cache_control": control}
                    ],
                },
                {
                    "type": "tool_use",
                    "id": "u",
                    "name": "n",
                    "input": {"cache_control": "kept: a tool's input"},
                },
            ],
        }
        kept = module.kept("messages", message)
        self.assertNotIn("cache_control", kept["content"][0])
        self.assertNotIn("cache_control", kept["content"][1])
        self.assertNotIn("cache_control", kept["content"][1]["content"][0])
        self.assertIn("cache_control", kept["content"][2]["input"])
        tool = module.kept("tools", {"name": "n", "cache_control": control})
        self.assertEqual(tool, {"name": "n"})
        members = [
            definition.get("properties", {})
            for definition in module.definitions().values()
        ]
        self.assertFalse(any("cache_control" in each for each in members))

    # kept.2: blocks of any depth of nesting are cleared of breakpoints.
    def test_any_depth(self):
        block = {"type": "text", "text": "x", "cache_control": {}}
        for _ in range(3000):
            block = {"type": "tool_result", "content": [block], "cache_control": {}}
        kept = module.kept("tools", block)
        for _ in range(3000):
            self.assertNotIn("cache_control", kept)
            kept = kept["content"][0]
        self.assertEqual(kept, {"type": "text", "text": "x"})

    # schema.1, schema.7: the core reads the schema whole, and a
    # recorded message and a string system prompt validate.
    def test_valid(self):
        root = {"title": "t", "definitions": module.definitions()}
        root["properties"] = module.units()
        root = _schema.read(_json.encode(root))
        units = module.units()
        message = _json.decode(
            b'{"role":"user","content":[{"type":"text","text":"hi"},'
            b'{"type":"tool_addition","tool":{"type":"tool_reference","name":"x"}}]}'
        )
        self.assertTrue(_json_schema.validates(message, units["messages"], root))
        self.assertTrue(_json_schema.validates("be brief", units["system"], root))

    # schema.7, schema.8: a type the package does not export is named by
    # its module, and each definition cites its file.
    def test_names(self):
        names = module.definitions()
        self.assertIn("BetaMessageParam", names)
        self.assertIn("beta_tool_use_block.Caller", names)
        self.assertEqual(
            names["BetaMessageParam"]["$comment"],
            "types/beta/beta_message_param.py › BetaMessageParam",
        )


if __name__ == "__main__":
    unittest.main()
