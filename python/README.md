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

A file is read by its extension. A directory is read by the product
whose directory it is, as Claude Code's raw API bodies (below):

```
living-memory path/to/dir
```

A file that is not OTLP JSON Lines stops the conversion, and the message
names the line by its number, the first line being line 1: a line that
is not JSON, a blank line, a byte order mark, a value not in the OTLP
JSON encoding, a key repeated in one list of attributes, an attribute
whose key is empty, more than one kind of data in the file, or one of
the eight that does not validate against its schema. What the file
holds but the sheets do not carry is reported as a warning: a member
with an unknown name, a field used only by Profiling, a MetricsData
line, and a field MTSV cannot represent.

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
as a warning on the `living_memory._otlp_json` logger, the record's
`left_behind` attribute holding the names. Each module logs on the logger
of its own name, under `living_memory`, and no handler is attached.

## Read Claude Code's raw API bodies

With `OTEL_LOG_RAW_API_BODIES=file:<dir>`, Claude Code writes its raw
API bodies, the Messages API request and response of every successful
call, into `<dir>`, with an index file, `index.jsonl`. The reader of
Claude Code, a product of the provider Anthropic, reads that directory,
from the command (`living-memory path/to/dir`) or in Python:

```python
from living_memory.providers.anthropic.claude_code import raw_api_bodies

sheets = raw_api_bodies.load("path/to/dir")
```

Each call becomes one OpenTelemetry GenAI event, read into the same
sheets as an OTLP file. `raw_api_bodies.logs_data` returns the events
themselves. Raw API bodies that do not conform raise
`raw_api_bodies.RawAPIBodiesDecodeError`, a `ValueError` whose `lineno`
is the line of the index file; a file the index names but that is not
on disk is logged as a warning on the `raw_api_bodies` module's logger.
What is read and written, and what is absent, is stated beside each
module, in the provider's folder
`providers/anthropic`: `messages.mtsv` for the Messages API request and
response, `claude_code/raw_api_bodies.mtsv` for Claude Code's index file.

## Keep Claude Code's conversations

From a clone, at the root of the repository, install living-memory in an
environment of its own, then install it into Claude Code:

```
pipx install ./python
living-memory --install=claude-code
```

The second command states what it will change and asks first; nothing
changes unless you answer `y`. It creates the data directory,
`~/.local/share/living-memory` on Linux and `~/Library/Application
Support/living-memory` on macOS, installs the Claude Code plugin
`claude-code` from this repository's marketplace, and last sets
`OTEL_LOG_RAW_API_BODIES` in `~/.claude/settings.json`. Setting it is
consent: Claude Code then saves every Messages API request and response,
each request holding the whole conversation so far, into the data
directory's `anthropic/claude_code/raw_api_bodies`, named after the module
that reads them. From the next session, after each response the plugin
converts them to `raw_api_bodies.mtsv` beside it, and at the start of each
session it tells Claude where that file is. Claude Code does not run the
conversion when you interrupt a response, so that turn's calls are
converted after the next response. Each conversion writes the
whole file again: an MTSV file is a sequence of sheets, so a new call's
rows, which belong to many sheets, are not appended as a JSON Lines file's
line is. What the installer and the plugin do is stated in
`claude_code/install.mtsv`.

To remove it all: `claude plugin uninstall claude-code@living-memory`,
delete `OTEL_LOG_RAW_API_BODIES` from `env` in `~/.claude/settings.json`,
and delete the data directory.

## Layout

Each module hides one decision, named beside it; a module uses only the
modules above it.

```
src/living_memory/
  _json                  level 1   how a JSON text is read and written, and a value's type (RFC 8259)
  _json_pointer          level 1   how a JSON Pointer is written and evaluated (RFC 6901)
  _fields                level 1   how a value is written as the text of an MTSV field
  _json_lines            level 1   how a JSON Lines file is split into its lines
  _json_schema           level 2   whether a value validates against a JSON Schema of draft-07
  _protojson             level 2   how a simple value of an OTLP message is written in JSON
  _relations             level 2   how a value becomes keyed sheets (Codd's normal form)
  _any_value             level 3   how an AnyValue and the JSON value it maps to are converted
  _parts                 level 3   the message part types, shared by the schemas that repeat them
  _system_instructions   level 4   the system instructions schema
  _tool_definitions      level 4   the tool definitions schema
  _input_messages        level 4   the input messages schema
  _output_messages       level 4   the output messages schema
  _tool_call_arguments   level 4   the tool call arguments schema
  _tool_call_result      level 4   the tool call result schema
  _memory_records        level 4   the memory records schema
  _retrieval_documents   level 4   the retrieval documents schema
  _event                 level 4   how a call is written as the GenAI inference details event
  _sets                  level 5   which attributes are the eight, and how one of them is read
  _telemetry_data        level 6   the OTLP data tree
  _otlp_json             level 7   how an OTLP JSON Lines file is read: the core's reader
  providers/             level 8
    __init__             which providers' products exist, by the file a product's directory holds and by plugin name
    anthropic/
      messages           how a Messages API call becomes the attributes of an event, as messages.mtsv states
      claude_code/
        raw_api_bodies   how Claude Code's raw API bodies are read, as raw_api_bodies.mtsv states
        install          what Claude Code must be told, as a plan of steps, as install.mtsv states
  _command               level 9   how a person runs a conversion, or carries out a plugin's plan
  __main__               level 10  the command's entry point
  __init__                         the public interface: load, loads, OTLPDecodeError
tests/                   one file per module, and the conformance runner
```

A level is one above the highest level of the modules it uses.

## Test

From the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
.venv/bin/python -m unittest discover -s python/tests
```

## License

[MIT](https://github.com/demos-ra/living-memory/blob/main/LICENSE)
