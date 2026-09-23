# living-memory: the eight GenAI text sets as MTSV

This document states how the eight structured text sets of the
OpenTelemetry GenAI semantic conventions [OTEL-GENAI] are written as
Multi-Sheet Tab-Separated Values, draft-demosra-mtsv-01 [MTSV]. Each rule
is numbered; the conformance folder of the same number checks it.

The eight sets are the attributes `gen_ai.system_instructions`,
`gen_ai.tool.definitions`, `gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `gen_ai.memory.records` and
`gen_ai.retrieval.documents` ([OTEL-GENAI], gen-ai-spans: Inference,
Execute tool span, Memory, Retrievals).

## M1. The input

The input is an OTLP JSON Lines file [OTEL-FILE-EXPORTER]: UTF-8, one
JSON value per line, each line a `TracesData`, `MetricsData` or
`LogsData` in OTLP JSON encoding, one kind of data per file (JSON File
serialization). The eight are recorded on spans and on events
([OTEL-GENAI], gen-ai-spans: Inference), so `TracesData` and `LogsData`
carry them and `MetricsData` carries none.

## M2. Order of lines

The lines of a file are in no guaranteed order ([OTEL-FILE-EXPORTER],
Streaming appending). They are read in the file's order, and nothing is
sorted.

## M3. Unknown fields

A receiver ignores fields with unknown names ([OTLP], JSON Protobuf
Encoding).

## M4. Identifiers

Trace and span ids are hex strings ([OTLP], JSON Protobuf Encoding),
case-insensitive ([RFC4648], Section 8). They are written as the text
the input holds, and compared without regard to case.

## M5. Finding the eight

Each of the eight is found by its attribute key. On events it is
structured; on spans it is structured, or a JSON string where structure
is not supported ([OTEL-GENAI], gen-ai-spans: Inference). Both forms are
read.

## M6. The eight apart

Each of the eight has sheets of its own. Input messages and output
messages are kept apart, each with its own sheets.

## M7. One sheet per definition

Every row of a relation draws on the same domains ([CODD1970], Section
1.3). A schema definition with fields of its own ([OTEL-GENAI],
model/gen-ai) is therefore a sheet of its own: each part type, each tool
type and each record type.

## M8. Nested fields

A nonsimple field is struck from its parent and becomes a child sheet
that carries its parent's key ([CODD1970], Section 1.4, normalization).
Normalization applies because the nesting is a tree and no key is
nonsimple ([CODD1970], Section 1.4).

## M9. The key of every row

Every row of the eight carries the `traceId` and `spanId` of the span or
event it came from: a span is identified by its trace id and span id
([OTLP], trace.proto), and normalization copies the parent's key down
([CODD1970], Section 1.4).

## M10. Values of any shape

A value of any shape is one sheet whose rows refer to their parent row
in the same sheet; a foreign key may refer to its own relation
([CODD1970], Section 1.3).

## M11. Order

The rows of a relation are unordered ([CODD1970], Section 1.3), while a
JSON array is an ordered sequence ([RFC8259], Section 5) and input
messages are in the order sent ([OTEL-GENAI], gen-ai-spans: Inference).
Order is therefore kept as position columns in the key.

## M12. Enums

An enum (`role`, `modality`, `finish_reason`) holds a simple, atomic
value ([CODD1970], Section 1.3), so it is a column.

## M13. Further properties

Every object of the eight allows further properties ([OTEL-GENAI],
model/gen-ai). A property beyond its schema goes to its object's value
sheet (M10).

## M14. Numbers

An implementation may limit the range and precision of numbers
([RFC8259], Section 6). A number is written as the input writes it.

## M15. Values MTSV cannot represent

A field that holds HT, LF, FF or CR cannot be represented, and a
generator does not write it ([MTSV], Generators). Such a value is not
written.

## M16. Names

Open. Sheet and column names are the sources' own; the names of the key
and position columns are not yet set.

## M17. Log records without ids

Open. A log record carries a trace id and span id only optionally
([OTLP], logs.proto); the key of one without them is not yet set.

## References

[CODD1970] Codd, E., "A Relational Model of Data for Large Shared Data
Banks", Communications of the ACM 13(6), 377-387, June 1970.

[MTSV] Ra, D., "Multi-Sheet Tab-Separated Values (MTSV)", Work in
Progress, Internet-Draft, draft-demosra-mtsv-01.

[OTEL-FILE-EXPORTER] OpenTelemetry, "OpenTelemetry Protocol File
Exporter".

[OTEL-GENAI] OpenTelemetry, "OpenTelemetry GenAI semantic conventions".

[OTLP] OpenTelemetry, "OpenTelemetry Protocol Specification" and its
Protobuf definitions.

[RFC4648] Josefsson, S., "The Base16, Base32, and Base64 Data
Encodings", RFC 4648, October 2006.

[RFC8259] Bray, T., Ed., "The JavaScript Object Notation (JSON) Data
Interchange Format", STD 90, RFC 8259, December 2017.
