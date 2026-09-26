# living-memory

living-memory converts JSON into Multi-Sheet Tab-Separated Values (MTSV).
A JSON Schema describes the JSON: each kind of thing it names becomes one
sheet, a simple member that one instance holds once becomes a column, and
what an instance can hold many of becomes a sheet of its own; values of mixed
type share one sheet of instances, and every row carries its owner's pointer
and its own, the keys that relate the sheets.
Every value is written once, and text holding tabs, line breaks or form
feeds is laid out by page, line and position. The module that reads a
source supplies its schema, so the conversion names no source.

The first integration reads Claude Code's recording of its API calls,
keeping each piece of a conversation once; `living-memory
--install=claude-code` turns the recording and the conversion on.

* [MTSV specification](https://github.com/demos-ra/mtsv-spec)
* [MTSV implementation](https://github.com/demos-ra/mtsv)

## Layout

| Folder         | Contents                                                |
|----------------|---------------------------------------------------------|
| `spec/`        | The specification, as MTSV: its rules and the fields it writes |
| `conformance/` | Test files shared by every implementation               |
| `python/`      | The Python implementation                               |

Conformance is the same for every language, so it sits beside the language
folders rather than inside one.

## Conformance

See [conformance/README.md](conformance/README.md).

## Python

See [python/README.md](python/README.md), which says how to install it. Its
version is the `version` field of
[python/pyproject.toml](python/pyproject.toml); there is no release yet.

## Help

living-memory is maintained by Demos Ra.

## License

[MIT](LICENSE)
