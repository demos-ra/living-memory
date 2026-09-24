# living-memory

living-memory writes the text around an AI model as Multi-Sheet
Tab-Separated Values (MTSV). An OpenTelemetry Protocol (OTLP) JSON Lines
file of traces or logs becomes MTSV sheets, whole: its envelopes,
resources, scopes, spans, log records and attributes, and the eight
structured text sets of the OpenTelemetry GenAI semantic conventions —
system instructions, tool definitions, input messages, output messages,
tool call arguments, tool call result, memory records and retrieval
documents — in sheets of their own, nested values becoming sheets of
their own in normal form.

* [MTSV specification](https://github.com/demos-ra/mtsv-spec)
* [MTSV implementation](https://github.com/demos-ra/mtsv)

## Layout

| Folder            | Contents                                             |
|-------------------|------------------------------------------------------|
| `spec/`           | The mapping, as MTSV: its rules and its sheets       |
| `conformance/`    | Test files shared by every implementation            |
| `python/`         | Python implementation                                |
| `plugins/`        | A plugin for each product it installs into          |
| `.claude-plugin/` | The catalog Claude Code reads the plugins from       |

The specification states the mapping; the conformance files check an
implementation against it; each language folder holds one implementation.
Conformance is the same for every language, so it sits beside them rather
than inside one. A provider's own format, and each of its products', is
read by an integration of its own, and the mapping names no provider. A
product that records conversations has a plugin of its own, which runs
the conversion there; Claude Code requires its catalog at the root of the
repository.

`spec/living-memory.mtsv` states the mapping: its conformance clause,
each rule, each sheet with its fields, and the references it cites.

## Conformance

See [conformance/README.md](conformance/README.md).

## Python

See [python/README.md](python/README.md). Its version is the `version`
field of [python/pyproject.toml](python/pyproject.toml).

## Status

In construction; there is no release yet. Versions will follow
[Semantic Versioning](https://semver.org).

## Help

living-memory is maintained by Demos Ra.

## License

[MIT](LICENSE)
