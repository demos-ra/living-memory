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
living-memory [-k] [-f READER] [-o OUTPUT] INPUT [OUTPUT]
living-memory [-f READER] [-o OUTPUT] --names INPUT
living-memory [-f READER] [-o OUTPUT] --filter=NAME [--values=FIRST-LAST] [--places=PLACE;...] INPUT
living-memory [-f READER] [-o OUTPUT] --new=NAME [--places=PLACE;...] INPUT
living-memory [-k] [-f READER] --add-context=HOST INPUT
living-memory --install=HOST
```

`INPUT` is read by the integration `-f`, or `--from`, names, by its path
among the integrations, as `anthropic/claude_code/raw_api_bodies`; else
a file is read by the integration of its extension, and a directory by
the integration of the file it holds. The output is a data bank, named
`INPUT` with its extension replaced by `.mtsv`, beside it, unless named.
Each input is stored apart, named as its integration names it,
`DATE/SESSION` for Claude Code's recording, else by the input file's
name without its extension; a conversion stores only the values the
data bank does not yet hold, each whole, one conversion at a time. A
source of several inputs is read line by line, and only the lines not
yet read. `-` is standard output, which gets the whole MTSV file.

Once stored, the source's files that its integration names as spent are
removed; `-k`, or `--keep-files`, keeps them.

What a data bank stores is given only by what it communicates, as MTSV
on standard output: `--names`, each input with the number of its values,
each sheet by its place and name, and each field of its header by its
position; `--filter=NAME`, an input's records, of the values at the
positions `--values` gives, as `3` or `2-5`, both ends included, and of
the sheets at the places `--places` gives, as `0;3`, each counted from 0;
and `--new=NAME`, what is new of an input since it was last asked, of
the places `--places` gives.

`--add-context=HOST` stores what is new, then reads the host's hook
input on standard input and writes to standard output the context that
host's integration composes of what the data bank communicates. With
it, every failure exits with status 1, never 2, which a hook reads as a
blocking error.

The command runs on Linux and macOS: one conversion at a time is kept by
a POSIX file lock, which Windows does not have. The library runs on any
platform. `--install=HOST` states the change an integration makes to a
host's configuration and asks first.

## Layout

Each module hides one decision, named beside it: a source the rules
follow, or one group of the specification's rules; a module uses only
the modules below it. The integrations are part of the package, and use
the core's modules and `_rename` as the command does.

```
src/living_memory/
  core
    _order          level 1   the order of sheets, records and columns: order.1-3
    _separators     level 1   which characters separate MTSV text, and what a field cannot hold (MTSV)
    _utf_8          level 1   how an octet sequence is read as UTF-8 (RFC 3629)
    _json_pointer   level 2   how a JSON Pointer is written, read from a URI fragment and evaluated, and a place named (RFC 6901)
    _json           level 3   how JSON texts are read, JSON strings written, and values typed (RFC 8259)
    _key            level 3   how each record is identified: key.1-5
    _field          level 4   how a value is written as a field's text, and what is not carried: field.1-5
    _json_schema    level 4   whether a JSON value validates against a JSON Schema of draft-07
    _storage        level 4   what a data bank holds of an input: the records of its values stored whole: storage.4
    _schema         level 5   what a module specification supplies, and how its schema is read: schema.1-15
    _value          level 5   how each input value is read, and when it is rejected: value.1-10
    _communication  level 5   what a data bank communicates: the names, and records by their keys: communication.1-5
    _relation       level 6   which relation and column each value is written to: relation.1-17
    _record         level 7   each instance as one record of its sheet: record.1
    _sheet          level 7   the name and the header of each sheet: sheet.1-5
    _file           level 8   how the whole input is written as one MTSV file: file.1-3
    _converter      level 9   the converter: the input and its module specification, as MTSV
    __init__        level 10  the public interface: convert, sheets, NonConformingError, NonConformingInputError
  shared by the integrations and the command
    _rename         level 1   how a file is replaced whole: written beside its name, renamed onto it
  integrations/               the set of integrations, each found as the package holds it
    anthropic/messages        the schema of what a model is given and generates, from the SDK's beta types
    anthropic/claude_code/raw_api_bodies
                              Claude Code's recording as input values, one input per session, one value
                              per request, only what the request it extends does not hold, only the
                              lines not yet read; the files it has spent
    anthropic/claude_code/install
                              what Claude Code is told: to record, to run the conversion, and the context
                              it composes of what the data bank communicates
  command
    _store          level 5   the data bank's storage, its secret: each input's sheets as files,
                              appended to, and cut back to the values stored whole after a conversion
                              cut short, the inputs in the order stored, how many values are stored
                              whole and communicated, and how many lines of the source are read:
                              storage.1-6
    _command        level 11  the command: a source read by its integration, stored in a data bank,
                              and communicated: communication.6
tests/                        one file per module, and the conformance runner, which reads
                              ../conformance, a data bank's cases through the command, and so
                              runs from a clone
tools/anthropic_schema.py     generates anthropic/messages.schema.json from the SDK's
                              type files, at the commit messages.mtsv cites
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
