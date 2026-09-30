# Conformance

Every implementation is tested against the same files. Each case checks
a requirement of `spec/living-memory.mtsv`, a rule, and is named for it:
the rule's Id, then what it asserts, as `relation.14.runs`. Every
expected file is derived from the rules by hand, never from an
implementation's output. The specification defines two classes of
product (conformance.1): a converter, tested by the first three folders,
and a data bank, tested by `communicated/`, only by what it communicates
(conformance.2, storage.6).

## Files of a case

A converter's case:

| File               | Holds                                                    |
|--------------------|----------------------------------------------------------|
| `name.jsonl`       | the input values, in order, one JSON text per line       |
| `name.schema.json` | the schema the module specification supplies (schema.1)  |
| `name.mtsv`        | the expected file, the caller naming it `name` (file.1)  |

A data bank's case holds its inputs and schema as a converter's does,
each input named by its `.jsonl` file without the extension (storage.2),
and one expected file for each communication its steps list below. An
expected file with no sheet is empty, a file of no bytes.

## Folders, by what an implementation does with each case

| Folder                   | Holds                                                  | Rules                          |
|--------------------------|--------------------------------------------------------|--------------------------------|
| `conforming/`            | cases converted whole                                  | conformance.2                  |
| `cannot-be-represented/` | cases with text not carried, the pointers listed below | conformance.2, field.4, field.5 |
| `non-conforming/`        | cases rejected, as listed below                        | conformance.2, schema.8, value.7-10 |
| `communicated/`          | cases stored by a data bank and communicated           | conformance.2, storage.1-6, communication.1-6 |

Every conforming case of more than one value is also converted in two
parts, its values before the middle, then the rest, and each part's
sheets appended sheet by sheet in the file's order give the same file
(value.2).

## A data bank's steps

Each case begins with a data bank that holds nothing. A range of
positions includes its first and its last (communication.4).

`storage.3.parts`, one input stored in two parts:

1. Store values 0-1.
2. Communicate the names: `storage.3.parts.names.mtsv`.
3. Communicate what is new: `storage.3.parts.new-1.mtsv`.
4. Store values 2-3.
5. Communicate what is new: `storage.3.parts.new-2.mtsv`.
6. Communicate what is new again: `storage.3.parts.again.mtsv`.
7. Request every sheet for every value stored: `storage.3.parts.all.mtsv`.
8. Request values 1-2, the sheets at places 1 and 2: `storage.3.parts.request.mtsv`.

`storage.2.two-inputs`, two inputs, each stored whole:

1. Store `storage.2.two-inputs.a`, then `storage.2.two-inputs.b`.
2. Communicate the names: `storage.2.two-inputs.names.mtsv`.
3. Request every sheet of each input for every value stored:
   `storage.2.two-inputs.a.all.mtsv`, `storage.2.two-inputs.b.all.mtsv`.

`storage.4.rejected-part`, a part rejected:

1. Store values 0-1.
2. Store values 2-3: rejected, as listed below.
3. Request every sheet for every value stored: `storage.4.rejected-part.all.mtsv`.

## Rules and their cases, in the spec's order

A case named for another rule also checks the rule it is listed under.

| Rule            | Cases |
|-----------------|-------|
| conformance.1   | every case: a converter's in the first three folders, a data bank's in `communicated/` |
| conformance.2   | every case |
| conformance.3   | no case: the key words' meaning |
| conformance.4   | no case: the order of the rules |
| conformance.5   | no case: the terms' definitions |
| schema.1        | non-conforming `schema.1.not-json`, `schema.1.not-draft-07`; any size or depth: every case, not tested separately |
| schema.2        | `schema.2.title`; non-conforming `schema.2.no-title` |
| schema.3        | non-conforming `schema.3.title` (LF), `schema.3.property` (HT), `schema.3.pattern` (FF), `schema.3.dependency` (CR), `schema.3.ref` (a $ref's last token, HT) |
| schema.4        | `schema.4.tokens` (every token of the subset; a pattern not anchored); non-conforming `schema.4.lookahead` |
| schema.5        | non-conforming `schema.5.outside`, `schema.5.missing` |
| schema.6        | non-conforming `schema.6.plain-name`, `schema.6.character` |
| schema.7        | non-conforming `schema.7.all-of`, `schema.7.any-of`, `schema.7.one-of`, `schema.7.not`, `schema.7.if`, `schema.7.then`, `schema.7.else`, `schema.7.dependency`; `relation.1.root-kind`, `relation.2.one-sheet` (a $ref reaching its schema again at a child location) |
| schema.8        | every non-conforming case of schema.1-7 |
| schema.9        | every conforming case |
| schema.10       | `schema.10.ref` (other members ignored, the fragment percent-decoded, a kind) |
| schema.11       | `schema.11.order` (every keyword that shapes a sheet at an object, written in reverse; two anyOf branches in the order written); `relation.13.arrays` (items, then additionalItems) |
| schema.12       | non-conforming `value.9.*`, one case for each assertion |
| schema.13       | `schema.13.no-sheet` (not, propertyNames, definitions, contains) |
| schema.14       | `schema.14.not-asserted` (format, contentMediaType, contentEncoding) |
| schema.15       | `schema.15.ignored` ($id, an unknown keyword) |
| value.1         | `value.1.order` |
| value.2         | every conforming case of more than one value, converted in two parts |
| value.3         | every case, not tested separately: the rule sets no limit |
| value.4         | non-conforming `value.4.byte-order-mark` |
| value.5         | `relation.15.molten` (members numbered in the order written) |
| value.6         | `field.1.simple` (a number as written; an integer a number) |
| value.7         | non-conforming `value.7.not-json`, `value.7.invalid-utf-8` |
| value.8         | non-conforming `value.8.duplicate-names` |
| value.9         | non-conforming `value.9.type`, `value.9.enum`, `value.9.const`, `value.9.multiple-of`, `value.9.maximum`, `value.9.exclusive-maximum`, `value.9.minimum`, `value.9.exclusive-minimum`, `value.9.max-length`, `value.9.min-length`, `value.9.pattern`, `value.9.max-items`, `value.9.min-items`, `value.9.unique-items`, `value.9.max-properties`, `value.9.min-properties`, `value.9.required`, `value.9.dependencies` |
| value.10        | non-conforming `value.10.deepest`, and every rejected value's place below |
| relation.1      | `relation.1.root-kind`; every conforming case |
| relation.2      | `relation.2.one-sheet` (a kind referred to twice and holding itself, one sheet), `relation.2.same-location` (anyOf, oneOf, if, then, else, a dependency); `relation.1.root-kind`, `schema.10.ref`, `schema.15.ignored` |
| relation.3      | `relation.3.all-of-ref` |
| relation.4      | `relation.4.narrowed` (allOf narrows; a type list; the union of anyOf's branches) |
| relation.5      | every conforming case |
| relation.6      | `field.1.simple` (string, number, boolean, null) |
| relation.7      | `schema.11.order` (string), `relation.13.arrays` (number, boolean), `relation.9.root-array` (null) |
| relation.8      | `relation.8.own` (a required object, nested; allOf) |
| relation.9      | `relation.9.root-array` (items); `schema.11.order` (an array property, patternProperties, additionalProperties); `relation.13.arrays` (items as an array, additionalItems) |
| relation.10     | `relation.10.optional` |
| relation.11     | `schema.11.order` (anyOf, oneOf, if, then, else, a dependency; a branch an instance is not valid against); `relation.4.narrowed`, `relation.16.once` |
| relation.12     | `relation.12.branches` (a location of several types: a string molten, an object by its branch; a location of one type, object, its own schema giving nothing) |
| relation.13     | `relation.13.arrays` (an element that is an array; items as an array, additionalItems) |
| relation.14     | `relation.14.runs` (HT, CRLF, FF; empty runs; two strings, one sheet of runs); `relation.15.molten` |
| relation.15     | `relation.15.molten` (a property of every type, an optional string, a member additionalProperties omitted takes), `relation.15.root-molten` (the input values, the sheet of the input values the sheet of instances); `relation.4.narrowed` (a type list), `relation.12.branches` (a branch of one simple type, taken as not required) |
| relation.16     | `relation.16.once` (a member two branches name, a member a branch's pattern covers, additionalProperties taking only the rest); `schema.11.order` |
| relation.17     | `relation.17.all-of` (no allOf column for a member named before); `relation.16.once` (a later sheet's field for a member written earlier, empty) |
| key.1           | every expected file |
| key.2           | every expected file's sheet of the input values; `relation.1.root-kind` (a kind's sheet), `relation.15.root-molten` (the sheet of instances) |
| key.3           | every other sheet; `relation.2.same-location`, `schema.11.order` (a record of its location's own instance), `key.3.own-branches` (a location relation.8 writes as its instance's own); `relation.2.one-sheet` (a parent in the same sheet); `relation.13.arrays` (a parent that is not the input value) |
| key.4           | `key.4.escaped` ('~1', '~0') |
| key.5           | `relation.14.runs`, `relation.15.molten` |
| order.1         | `schema.11.order`, `relation.13.arrays` (a sheet followed by its subordinate sheets), `relation.2.one-sheet` (a kind at its first reference), `relation.15.molten` (the sheet of instances, then the sheet of runs, last) |
| order.2         | `value.1.order`, `relation.2.one-sheet`, `relation.15.molten` |
| order.3         | `relation.8.own` (the schema's order, not the input's; a member in its property's place; allOf's last) |
| file.1          | every case: one file, named by the caller |
| file.2          | `file.2.empty-sheet`; every expected file |
| file.3          | every expected file |
| sheet.1         | `schema.2.title`; `relation.1.root-kind` (a kind's sheet), `relation.15.root-molten` (the sheet of instances) |
| sheet.2         | `relation.2.one-sheet` (a title), `schema.10.ref` (a $ref's last token, decoded), `relation.2.same-location` |
| sheet.3         | `sheet.3.doubled` (a title, a property's name, and a name composed of them, beginning with '_'); `relation.14.runs` (`_runs`), `relation.15.molten` (`_instances`) |
| sheet.4         | `sheet.4.names` (patterns, a dependency's key, positions, a keyword of one subschema); `schema.11.order` (titles), `relation.8.own` (a property relation.8 writes, in the name), `relation.12.branches`, `relation.13.arrays` (positions, items) |
| sheet.5         | every expected file; `relation.8.own` (property.member) |
| record.1        | every expected file |
| field.1         | `field.1.simple`, `relation.15.molten` |
| field.2         | `relation.15.molten` (all six types) |
| field.3         | `relation.14.runs`, `relation.15.molten` |
| field.4         | cannot-be-represented `field.4.not-carried` (a lone CR, an unpaired surrogate, HT, LF, FF and CR in member names) |
| field.5         | cannot-be-represented `field.4.not-carried` |
| storage.1       | `storage.3.parts` (`all`: part 1's records unchanged, part 2's added) |
| storage.2       | `storage.2.two-inputs` |
| storage.3       | `storage.3.parts` (`all` is the input's file for the values stored) |
| storage.4       | `storage.4.rejected-part` (nothing of part 2 stored, value 2 included) |
| storage.5       | `storage.3.parts` (part 2's positions begin at 2) |
| storage.6       | every case of `communicated/`: nothing is read but what is communicated |
| communication.1 | every expected file of `communicated/` |
| communication.2 | every expected file of `communicated/`: values as stored, records in stored order |
| communication.3 | `storage.3.parts.names` (the places of the sheets stored), `storage.2.two-inputs.names` (two inputs, in the order stored) |
| communication.4 | every `all`; `storage.3.parts.request` |
| communication.5 | `storage.3.parts.new-1`, `storage.3.parts.new-2`, `storage.3.parts.again` (nothing new: no sheet) |
| communication.6 | `storage.3.parts.new-2` (part 2 alone: part 1 was communicated), `storage.3.parts.again` |

## Expected reports and rejections

Values are counted from 0, as key.2 counts positions, where JSON Lines
calls the first value 1; pointers are written as JSON strings (field.5,
value.10).

| Case                         | Expected |
|------------------------------|----------|
| `field.4.not-carried`        | not carried: `"/0/t"`, `"/0/u"`, `"/0/a\tb"`, `"/0/c\nd"`, `"/0/e\ff"`, `"/0/g\rh"` |
| `schema.*`                   | rejected: the module specification |
| `value.4.byte-order-mark`    | rejected: value 1, pointer `""` |
| `value.7.not-json`           | rejected: value 1, pointer `""` |
| `value.7.invalid-utf-8`      | rejected: value 1, pointer `""` |
| `value.8.duplicate-names`    | rejected: value 0, pointer `"/b"` |
| `value.9.required`           | rejected: value 0, pointer `""` |
| `value.9.dependencies`       | rejected: value 0, pointer `""` |
| every other `value.9.*`      | rejected: value 0, pointer `"/n"` |
| `value.10.deepest`           | rejected: value 1, pointer `"/p/k"` |
| `storage.4.rejected-part`    | rejected: value 3, pointer `"/n"` |
