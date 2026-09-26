"""Tests of integrations.anthropic.messages: the kept units' schema."""

import unittest

from living_memory import _json, _json_schema, _module
from living_memory.integrations.anthropic import messages as module


class TestUnits(unittest.TestCase):
    # kept.1: the four kept parts, each by its unit.
    def test_kept_parts(self):
        self.assertEqual(
            sorted(module.units()), ["content", "messages", "system", "tools"]
        )

    # schema.1, schema.3: the core reads the schema whole, and a
    # recorded message and a string system prompt validate.
    def test_valid(self):
        root = {"title": "t", "definitions": module.definitions()}
        root["properties"] = module.units()
        root = _module.read(_json.encode(root))
        units = module.units()
        message = _json.decode(
            b'{"role":"user","content":[{"type":"text","text":"hi"},'
            b'{"type":"tool_addition","tool":{"type":"tool_reference","name":"x"}}]}'
        )
        self.assertTrue(_json_schema.validates(message, units["messages"], root))
        self.assertTrue(_json_schema.validates("be brief", units["system"], root))

    # schema.3: a type the package does not export is named by its
    # module.
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
