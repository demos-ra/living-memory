# living-memory for Python

The Python implementation of living-memory: the eight structured text
sets of the OpenTelemetry GenAI semantic conventions, written as
Multi-Sheet Tab-Separated Values (MTSV). The version is the `version`
field of `pyproject.toml`.

In construction: the modules below hold their purpose and no code yet.

## Install

There is no release yet. From a clone, at the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
```

## Layout

Each module hides one decision, named beside it; a module uses only the
modules above it.

```
src/living_memory/
  _relations             how a value becomes keyed sheets
  _parts                 the message part types, shared by the schemas that repeat them
  _system_instructions   the system instructions schema
  _tool_definitions      the tool definitions schema
  _input_messages        the input messages schema
  _output_messages       the output messages schema
  _tool_call_arguments   the tool call arguments schema
  _tool_call_result      the tool call result schema
  _memory_records        the memory records schema
  _retrieval_documents   the retrieval documents schema
  integrations/
    __init__             which reader reads which input
    otlp_json            how an OTLP JSON Lines file is read, and where the eight are on its spans and events
    providers/__init__   which provider formats exist
  _command               how a person runs a conversion
  __main__               the command's entry point
  __init__               the public interface
tests/                   one file per module, and the conformance runner
```

## License

[MIT](LICENSE)
