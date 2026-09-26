# living-memory for Python

The Python implementation of living-memory: JSON that a JSON Schema
describes, converted to Multi-Sheet Tab-Separated Values (MTSV). The
version is the `version` field of `pyproject.toml`.

* [Specification](https://github.com/demos-ra/living-memory/blob/main/spec/living-memory.mtsv)
* [MTSV specification](https://github.com/demos-ra/mtsv-spec)

## Install

There is no release yet. From a clone, at the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
```

## Convert

```python
import living_memory

text = living_memory.convert(values, schema)
```

`values` are the input values, each a JSON text in UTF-8, in the order
their source holds them. `schema` is the JSON Schema that describes them,
itself a JSON text, its root schema holding a title. `convert` returns the
whole input as the text of one MTSV file, which the caller names; it holds
the sheets that hold a record.

An input can be converted in parts: `convert(values, schema, start)` gives
each value its position in the whole input, counted from `start`, and
`living_memory.sheets(schema)` names every sheet the schema gives, in the
file's order, so that the parts' sheets appended in that order are the
whole input's file.

A schema that does not conform raises `living_memory.NonConformingError`,
a `ValueError` whose `pointer` is the place within the schema. An
input value that does not conform raises its subclass
`living_memory.NonConformingInputError`, which adds `position`, the
value's position counted from 0, its `pointer` the place within the
value. What a field cannot hold is left out and logged as a warning on
the `living_memory._converter` logger, by its pointer; no handler is
attached.

## Command

```
living-memory [-o OUTPUT] INPUT [OUTPUT]
living-memory --install=HOST
```

`INPUT` is a file, read by the integration of its extension, or a
directory, read by the integration of the file it holds. The output is
`INPUT` with its extension replaced by `.mtsv`, beside it, unless named.
It is kept as a folder of its sheets' files, each only appended to: a
conversion adds the values the output does not yet hold, one conversion
at a time, and the files joined in the order their names give are the
MTSV file. `-` is standard output, which gets the whole MTSV file.

The command runs on Linux and macOS: one conversion at a time is kept by
a POSIX file lock, which Windows does not have. The library runs on any
platform. `--install=HOST` states the change an
integration makes to a host's configuration and asks first.

## Layout

Each module hides one decision, named beside it: a source the rules
follow, or one group of the specification's rules; a module uses only
the modules below it.

```
src/living_memory/
  _json_pointer   level 1   how a JSON Pointer is written and evaluated, and a place named (RFC 6901)
  _order          level 1   the order of sheets, records and columns: order.1-3
  _separators     level 1   which characters separate MTSV text, and what a field cannot hold (MTSV)
  _store          level 1   how the output is kept: its sheets' files, each only appended to
  _utf_8          level 1   how an octet sequence is read as UTF-8 (RFC 3629)
  _json           level 2   how JSON texts are read, JSON strings written, and values typed (RFC 8259)
  _key            level 2   how each record is identified: key.1
  _field          level 3   how a value is written as a field's text, and what is not carried: field.1-3
  _json_schema    level 3   whether a JSON value validates against a JSON Schema of draft-07
  _module         level 4   what a module specification supplies, and how its schema is read: module.1-2
  _value          level 4   how each input value is read, and when it is rejected: value.1-3
  _relation       level 5   which relation and column each value is written to: relation.1-5
  _record         level 6   each instance as one record of its sheet: record.1
  _sheet          level 6   the name and the header of each sheet: sheet.1-2
  _file           level 7   how the whole input is written as one MTSV file: file.1-3
  _converter      level 8   the converter: the input and its module specification, as MTSV
  __init__        level 9   the public interface: convert, NonConformingError, NonConformingInputError
  integrations/             the set of integrations, each found as the package holds it
    anthropic/messages      the schema of what a model is given and generates, from the SDK's beta types
    anthropic/claude_code/raw_api_bodies
                            Claude Code's recording as input values, each unit once, only what is new
    anthropic/claude_code/install
                            what Claude Code is told: to record, and to run the conversion
  _command        level 9   the command: a source read by its integration, written as MTSV
tests/                      one file per module, and the conformance runner, which reads
                            ../conformance and so runs from a clone
tools/anthropic_schema.py   generates anthropic/messages.schema.json from anthropic_sdk/,
                            the SDK's type files and their licence
```

## Test

From the root of the repository:

```
python3 -m venv .venv
.venv/bin/pip install ./python
.venv/bin/python -m unittest discover -s python/tests
```

## License

[MIT](https://github.com/demos-ra/living-memory/blob/main/LICENSE)
