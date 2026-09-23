# living-memory for Python

The Python implementation of living-memory: OpenTelemetry Protocol
(OTLP) JSON Lines files of traces and logs, and the eight structured text
sets of the OpenTelemetry GenAI semantic conventions among them, written
as Multi-Sheet Tab-Separated Values (MTSV). The version is the `version`
field of `pyproject.toml`.

## Install

There is no release yet. From a clone, at the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
```

## Convert a file

```
living-memory trace.jsonl
```

That writes `trace.mtsv` beside it. Name the output as an operand or
with `-o`, or `--output`; `-` is standard input or standard output:

```
living-memory trace.jsonl out.mtsv
living-memory -o out.mtsv trace.jsonl
living-memory - out.mtsv < trace.jsonl
```

A line that is not JSON stops the conversion, and the message names the
line by its number.

## Read and write in Python

```python
import living_memory
import mtsv

with open("trace.jsonl", "rb") as file:
    sheets = living_memory.load(file)

with open("trace.mtsv", "wb") as file:
    mtsv.dump(sheets, file)
```

`loads` works on a string. Every sheet of the mapping is returned, in the
order of the Sheets of `spec/living-memory.mtsv`, each with its header.

## Layout

Each module hides one decision, named beside it; a module uses only the
modules above it.

```
src/living_memory/
  _relations             level 1  how a value becomes keyed sheets
  _parts                 level 2  the message part types, shared by the schemas that repeat them
  _system_instructions   level 3  the system instructions schema
  _tool_definitions      level 3  the tool definitions schema
  _input_messages        level 3  the input messages schema
  _output_messages       level 3  the output messages schema
  _tool_call_arguments   level 3  the tool call arguments schema
  _tool_call_result      level 3  the tool call result schema
  _memory_records        level 3  the memory records schema
  _retrieval_documents   level 3  the retrieval documents schema
  _telemetry_data        level 4  the OTLP data tree, and where the eight are among its attributes
  integrations/          level 5
    __init__             which reader reads which input
    otlp_json            how an OTLP JSON Lines file is read: its lines, their order and index, and UTF-8
    providers/__init__   which provider formats exist
  _command               level 6  how a person runs a conversion
  __main__               level 6  the command's entry point
  __init__               the public interface: load, loads
tests/                   one file per module, and the conformance runner
```

## Test

From the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
.venv/bin/python -m unittest discover -s python/tests
```

## License

[MIT](LICENSE)
