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

files = living_memory.convert(values, schema, pointer)
for name, text in files.items():
    ...
```

`values` are the input values, each a JSON text in UTF-8, in the order
their source holds them. `schema` is the JSON Schema that describes them,
itself a JSON text, its root schema holding a title. `pointer` is the JSON
Pointer, as a JSON string, of the member whose value selects each value's
file; left empty, there is one file. `convert` returns a dict of each
file's name and its MTSV text; the one file of a conversion without a
pointer is named `""`, for the caller to name.

A schema or a pointer that does not conform raises
`living_memory.NonConformingError`, a `ValueError` whose `pointer` is the
place within the schema, `""` for a pointer that is not a JSON string. An
input value that does not conform raises its subclass
`living_memory.NonConformingInputError`, which adds `position`, the
value's position counted from 0, its `pointer` the place within the
value. What a field cannot hold is left out and logged as a warning on
the `living_memory._converter` logger, by its pointer; no handler is
attached.

## Layout

Each module hides one decision, named beside it; a module uses only the
modules below it.

```
src/living_memory/
  _json_pointer         level 1  how a JSON Pointer is written and evaluated, and a place named (RFC 6901)
  _fields               level 1  how text is written in MTSV fields, and what a field cannot hold
  _regular_expression   level 1  which regular expressions a schema may hold, and what they match
  _json                 level 2  how JSON texts are read, JSON strings written, and values typed (RFC 8259)
  _json_schema          level 3  whether a JSON value validates against a JSON Schema of draft-07
  _relations            level 4  how a schema and its instances become related sheets
  _converter            level 5  how the input becomes files
  __init__                       the public interface: convert, NonConformingError, NonConformingInputError
tests/                           one file per module, and the conformance runner,
                                 which reads ../conformance and so runs from a clone
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
