# living-memory

**Memory for AI systems, as plain tables that any tool, model or person can
read, and that any system can share.**

## The problem

Agents need memory that lasts, and teams need to share it. Today it is
usually summaries (detail lost), raw logs (repeated on every turn, locked to
one tool's format) or a database behind one app (closed to the next agent,
vendor or team).

## The framework

| Step          | What happens                                                                                   | Why                                                                                   |
|---------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **Keep**      | the full conversation, as the source sends it                                                  | nothing is summarized away                                                            |
| **Normalize** | each JSON tree becomes related tables, one per kind of thing, each fact once, each row linked to its parent (Codd's normal form) | no repeats; anything can be rebuilt exactly; stored data survives the source changing shape |
| **Store**     | in a data bank that only ever adds                                                             | history is never rewritten                                                            |
| **Share**     | on request: what is held, any part of it, or what is new                                       | an agent stays current by asking only for what changed; any system can consume it     |

## Why tables, not JSON

JSON is how systems *send* data. Tables are how data gets *used*.

|                  | JSON                                              | living-memory's tables (MTSV)                                   |
|------------------|---------------------------------------------------|-----------------------------------------------------------------|
| Shape            | nested, and it varies record to record            | fixed columns: one table per kind of thing                      |
| Repetition       | every record repeats its keys, and every request repeats the history | a header once per table; each fact once       |
| Finding things   | code has to walk each tree                        | filter, sort, count and join directly, by column                |
| Tools            | parsers and custom scripts                        | spreadsheets, `grep`, dataframes, SQL, language models          |
| Change over time | whole documents                                   | rows appended; changes show line by line                        |
| Formats          | one                                               | converts to and from CSV, Excel, ODS, JSON, SQLite, Parquet and Arrow |

[MTSV](https://github.com/demos-ra/mtsv-spec) is plain text: tab-separated
values, with several sheets in one file. It is already readable by the tools
you have, and [its library](https://github.com/demos-ra/mtsv) converts it to
and from the formats above.

## Why it is infrastructure

- **Specified.** A written specification and a shared conformance suite fix
  exactly what any input becomes, so any language, at any organization,
  produces the same tables.
- **Open at both ends.** Any JSON source comes in through its own
  integration; any system reads what comes out.
- **Yours.** Plain files on your disk.

## Get started

```
pipx install living-memory
```

In Python, JSON values and the JSON Schema that describes them in, one MTSV
file out:

```python
import living_memory

text = living_memory.convert(values, schema)
```

The `living-memory` command reads a source through its integration, stores it
in a data bank, and gives back what the data bank holds. See
[python/README.md](python/README.md) for the library, the command and each
integration.

## Integrations

Each integration reads one source, and adding one changes nothing else.

**Claude Code**, the first. It records Claude Code's requests and responses,
keeps each conversation, one data bank input per session, in
`~/Documents/living-memory/`, and gives Claude Code what the data bank holds
as context, at the start of each session and with each prompt:

```
living-memory --install=claude-code
```

It states the change it makes and asks first. It runs on Linux and macOS.

## What is in this repository

| Folder                           | Contents                                                              |
|----------------------------------|-----------------------------------------------------------------------|
| [`spec/`](spec/)                 | the specification, as MTSV: its rules and the fields it writes         |
| [`conformance/`](conformance/)   | the test files every implementation passes, in any language            |
| [`python/`](python/)             | the Python implementation: the library, the command and the integrations |

Conformance is the same for every language, so it sits beside the language
folders rather than inside one.

## Version

The version is the `version` field of
[python/pyproject.toml](python/pyproject.toml), 0.1.0. Each release is tagged
`vX.Y.Z` and published on [PyPI](https://pypi.org/project/living-memory/).

## Help

Ask a question or report a bug in
[the issues](https://github.com/demos-ra/living-memory/issues); report a
vulnerability as [SECURITY.md](SECURITY.md) says. living-memory is maintained
by Demos Ra.

## License

[MIT](LICENSE)
