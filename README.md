# living-memory

living-memory converts JSON into Multi-Sheet Tab-Separated Values (MTSV).
A JSON Schema describes the JSON: each object and array becomes a sheet of
its own, each simple member a column, and the sheets are related by keys.
Every value is written once, and text holding tabs, line breaks or form
feeds is laid out by page, line and position. The module that reads a
source supplies its schema, so the conversion names no source.

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
