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

A file is read by its extension. A directory is read by the provider
whose directory it is, as Claude Code's raw API bodies (below):

```
living-memory path/to/dir
```

A file that is not OTLP JSON Lines stops the conversion, and the message
names the line by its number, the first line being line 1: a line that
is not JSON, a blank line, a byte order mark, a value not in the OTLP
JSON encoding, a key repeated in one list of attributes, more than one
kind of data in the file, or one of the eight that does not validate
against its schema. What the file holds but the sheets do not carry is reported as a
warning: a member with an unknown name, a field used only by Profiling,
a MetricsData line, and a field MTSV cannot represent.

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
order of the Sheets of `spec/living-memory.mtsv`, each with its header. A
file that is not OTLP JSON Lines raises `living_memory.OTLPDecodeError`,
a `ValueError` whose `lineno` is the line. What is left behind is logged
as a warning on the `living_memory` logger, the record's `left_behind`
attribute holding the names.

## Read Claude Code's raw API bodies

With `OTEL_LOG_RAW_API_BODIES=file:<dir>`, Claude Code writes its raw
API bodies, the Messages API request and response of every successful
call, into `<dir>`, with an index file, `index.jsonl`. The `anthropic` provider reads
that directory, from the command (`living-memory path/to/dir`) or in
Python:

```python
from living_memory.integrations.providers import anthropic

sheets = anthropic.load("path/to/dir")
```

Each call becomes one OpenTelemetry GenAI event, read into the same
sheets as an OTLP file. `anthropic.logs_data` returns the events
themselves. What the provider reads and writes, and what is absent, is
stated in `src/living_memory/integrations/providers/anthropic.mtsv`.
Raw API bodies that do not conform raise
`anthropic.RawAPIBodiesDecodeError`, a `ValueError` whose `lineno` is the
line of the index file.

## Layout

Each module hides one decision, named beside it; a module uses only the
modules above it.

```
src/living_memory/
  _json                  level 1  how a JSON text is read, and the type of a JSON value
  _relations             level 2  how a value becomes keyed sheets
  _json_schema           level 2  whether a value validates against a JSON Schema of draft-07
  _protojson             level 2  how a simple value of an OTLP message is written in JSON
  _parts                 level 3  the message part types, shared by the schemas that repeat them
  _system_instructions   level 4  the system instructions schema
  _tool_definitions      level 4  the tool definitions schema
  _input_messages        level 4  the input messages schema
  _output_messages       level 4  the output messages schema
  _tool_call_arguments   level 4  the tool call arguments schema
  _tool_call_result      level 4  the tool call result schema
  _memory_records        level 4  the memory records schema
  _retrieval_documents   level 4  the retrieval documents schema
  _telemetry_data        level 5  the OTLP data tree, and where the eight are among its attributes
  integrations/          level 6
    __init__             which reader reads which input
    otlp_json            how an OTLP JSON Lines file is read: its lines, their order and number, UTF-8, what is rejected and what is reported
    providers/__init__   which providers exist, by the file a provider's directory holds
    providers/anthropic  how Claude Code's raw API bodies are read, as its module specification, providers/anthropic.mtsv, states
  _command               level 7  how a person runs a conversion
  __main__               level 7  the command's entry point
  __init__               the public interface: load, loads, OTLPDecodeError
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
