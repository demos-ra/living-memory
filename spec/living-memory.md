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
nonsimple ([CODD1970], Section 1.4). A child row's pointer (M9) extends
its parent's pointer, so the parent's key is carried in it.

## M9. The key of every row

Every row is keyed by three columns. `traceId` and `spanId` name the span
or event it came from: a span is identified by its trace id and span id
([OTLP], trace.proto). `pointer` is a JSON Pointer ([RFC6901]) from the
root of the attribute value to the row's value: `/0` for the first
message, `/0/parts/2` for its third part.

## M10. Values of any shape

A value of any shape is one sheet with a row per node: its `pointer`, its
`type`, one of `object`, `array`, `string`, `number`, `true`, `false`
and `null` ([RFC8259], Section 3), and its `value`.

## M11. Order

The rows of a relation are unordered ([CODD1970], Section 1.3), while a
JSON array is an ordered sequence ([RFC8259], Section 5) and input
messages are in the order sent ([OTEL-GENAI], gen-ai-spans: Inference).
A JSON Pointer names an array element by its zero-based index
([RFC6901], Section 4), so order is kept in the pointer.

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

Names are the sources' own: `traceId` and `spanId` as OTLP JSON writes
them ([OTLP], JSON Protobuf Encoding), `pointer` for the JSON Pointer
([RFC6901]), `type` and `value` for a JSON value ([RFC8259], Section 3),
and each schema's own field names ([OTEL-GENAI], model/gen-ai).

## M17. Log records without ids

Open. A log record carries a trace id and span id only optionally
([OTLP], logs.proto); the key of one without them is not yet set.

## M18. The output

The output is one MTSV file. A sheet is a header and zero or more
records, and an MTSV file is an ordered sequence of sheets ([MTSV], Data
Model): every sheet of the mapping is written, in the order of the
working record's Sheets, each with its header, and with no records where
the input holds none.

## M19. Null, empty and absent

Open. A field is text ([MTSV], Data Model), so an empty field does not
tell a null, an empty string and an absent field apart.

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

[RFC6901] Bryan, P., Ed., Zyp, K., and M. Nottingham, Ed., "JavaScript
Object Notation (JSON) Pointer", RFC 6901, April 2013.

[RFC8259] Bray, T., Ed., "The JavaScript Object Notation (JSON) Data
Interchange Format", STD 90, RFC 8259, December 2017.
