# Conformance

Every implementation is tested against the same files. Each case checks
one requirement of `spec/living-memory.mtsv`, a rule, and is named for
it: the rule's Id, then what it asserts, as `relation.14.text`. Every
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

A data bank's case, whose input is named by its `.jsonl` file, without
the extension (storage.2):

| File                  | Holds                                                     |
|-----------------------|-----------------------------------------------------------|
| `name.jsonl`          | the input values, in order, one JSON text per line        |
| `name.schema.json`    | the schema the module specification supplies (schema.1)   |
| `name.aggregate.mtsv` | the aggregate communicated at step 2 (communication.3)    |
| `name.new-1.mtsv`     | what is new, communicated at step 3 (communication.5)     |
| `name.new-2.mtsv`     | what is new, communicated at step 5 (communication.5)     |
| `name.again.mtsv`     | what is new, communicated again at step 6 (communication.6) |
| `name.all.mtsv`       | every record, requested at step 7 (communication.4)       |
| `name.request.mtsv`   | the records requested at step 8, where the case has one   |

An expected file with no sheet is empty, a file of no bytes.

## Folders, by what an implementation does with each case

| Folder                   | Holds                                                   | Rules                            |
|--------------------------|---------------------------------------------------------|----------------------------------|
| `conforming/`            | cases converted whole                                   | conformance.2                    |
| `cannot-be-represented/` | cases with text not carried, the pointers listed below  | conformance.2, field.4, field.5  |
| `non-conforming/`        | cases rejected, as listed below                         | conformance.2, schema.9, value.7-10 |
| `communicated/`          | cases stored by a data bank and communicated            | conformance.2, storage.1-6, communication.1-6 |

## A data bank's steps

Each case of `communicated/` runs these steps, in order, on a data bank
that holds nothing, the parts split as value.2's are: the values before
the middle, then the rest.

1. Store part 1.
2. Communicate the aggregate: `name.aggregate.mtsv`.
3. Communicate what is new: `name.new-1.mtsv`.
4. Store part 2.
5. Communicate what is new: `name.new-2.mtsv`.
6. Communicate what is new again: `name.again.mtsv`.
7. Request every sheet of the input for every value stored: `name.all.mtsv`.
8. Where the case has one, the request its row below states: `name.request.mtsv`.

A range of positions includes its first and excludes its last.
`storage.2.two-inputs` stores two inputs, `.a` then `.b`, each whole in
one part, then communicates the aggregate and, for each input, every
record: `name.aggregate.mtsv`, `name.a.all.mtsv`, `name.b.all.mtsv`. The
tests read nothing of a data bank's storage but what it communicates
(storage.6).

## Rules and their cases, in the spec's order

A case named for another rule also checks the rule it is listed under.

| Rule        | Cases                                                                 |
|-------------|-----------------------------------------------------------------------|
| schema.1    | non-conforming `schema.1.schema-not-json`, `schema.1.not-a-schema` (not a schema of draft-07) |
| schema.2    | `schema.2.title`; non-conforming `schema.2.no-title`                  |
| schema.3    | non-conforming `schema.3.property-name` (HT), `schema.3.title-name` (LF), `schema.3.pattern-name` (FF), `schema.3.dependency-name` (CR), `schema.3.ref-name` (a $ref's last token, HT) |
| schema.4    | `schema.4.pattern-subset` (every token of the subset); non-conforming `schema.4.pattern-token` |
| schema.5    | non-conforming `schema.5.ref-outside`, `schema.5.ref-leading-zero` (an array index with a leading zero references nothing) |
| schema.6    | non-conforming `schema.6.ref-plain-name` (a fragment that is not a JSON Pointer), `schema.6.ref-character` (a character a URI fragment does not hold) |
| schema.7    | non-conforming `schema.7.ref-root`                                    |
| schema.8    | non-conforming `schema.8.ref-loop-all-of`, `schema.8.ref-loop-any-of`, `schema.8.ref-loop-one-of`, `schema.8.ref-loop-not`, `schema.8.ref-loop-if`, `schema.8.ref-loop-then`, `schema.8.ref-loop-else`, `schema.8.ref-loop-dependency` (each keyword the loop may run through) |
| schema.9    | every non-conforming case of schema.1-8                               |
| schema.10   | every conforming case                                                 |
| schema.11   | `schema.11.ref`, `schema.11.ref-members-ignored`, `schema.11.ref-percent-encoded` (a fragment percent-decoded) |
| schema.12   | `order.1.keywords` (the order subschemas are taken)                   |
| schema.13   | non-conforming `value.9.schema-mismatch`, `value.9.maximum`           |
| schema.14   | `schema.14.no-sheet` (not, propertyNames, definitions), `schema.14.contains` |
| schema.15   | `schema.15.format-not-asserted`, `schema.15.content-not-asserted`     |
| schema.16   | `schema.16.ignored-keyword`                                           |
| value.1     | `value.1.order`                                                       |
| value.2     | every conforming case of more than one value, converted in two parts: its values before the middle, then the rest, each part's sheets appended in the file's order |
| value.3     | no case: the rule sets no limit, and every finite case lies within some limit, so no case can show it |
| value.4     | non-conforming `value.4.byte-order-mark`                              |
| value.5     | `value.5.member-order`                                                |
| value.6     | `value.6.number-as-written`, `value.6.integer`                        |
| value.7     | non-conforming `value.7.not-json`, `value.7.invalid-utf-8`            |
| value.8     | non-conforming `value.8.duplicate-names`                              |
| value.9     | non-conforming `value.9.schema-mismatch` (type), `value.9.maximum` (a numeric assertion) |
| value.10    | non-conforming `value.10.deepest-place`, and every rejected value's place below |
| relation.1  | every conforming case                                                 |
| relation.2  | `relation.2.kind-one-sheet` (one kind, referred to twice, one sheet), `relation.2.kind-holds-itself` (a kind whose rows key into its own sheet), `relation.2.kind-one-of`, `relation.2.kind-if`, `relation.2.kind-then`, `relation.2.kind-else`, `relation.2.kind-dependency` (a kind a subschema applies to the same location); `schema.11.ref`, `relation.12.any-of-types`, `relation.12.object-branches` (a kind placed branch by branch, by its branches' kinds) |
| relation.3  | `relation.3.all-of-ref` (a $ref that allOf applies: its members the instance's own, no kind's sheet) |
| relation.4  | `relation.4.all-of-narrows`; `relation.15.optional` (types listed), `relation.15.root-any` (none listed), `relation.11.any-of`, `relation.11.one-of` (the union of branches) |
| relation.5  | every conforming case                                                 |
| relation.6  | `schema.2.title`, `order.3.columns`                                   |
| relation.7  | `relation.9.pattern-properties`, `relation.9.additional-properties`, `key.4.escaped-name` |
| relation.8  | `relation.8.object` (a required object, nested), `relation.8.all-of`; `relation.4.all-of-narrows`, `relation.11.any-of`, `relation.11.one-of`, `relation.16.once` (a required object that allows one type, object, by allOf or the union of branches) |
| relation.9  | `relation.9.pattern-properties`, `relation.9.additional-properties`, `relation.9.root-array`; `schema.14.contains` |
| relation.10 | `relation.10.optional-object`                                         |
| relation.11 | `relation.11.any-of`, `relation.11.one-of`, `relation.11.if-then-else`, `relation.11.dependencies` |
| relation.12 | `relation.12.any-of-types` (a location of several types placed branch by branch: a string molten, an array by its elements, an object by its kind), `relation.12.any-of-in-items` (items whose own schema gives nothing: each element in its branch's sheet, no sheet of the items' own), `relation.12.object-branches` (the same for an optional property and for a kind, each instance in its branch's sheet or kind's) |
| relation.13 | `relation.13.array-of-arrays`, `relation.13.items-tuple` (items, additionalItems) |
| relation.14 | `relation.14.text` (a string's runs), `relation.14.empty-runs`, `relation.14.one-runs-sheet` (two strings' runs, one sheet of runs); `order.1.shared-last` |
| relation.15 | `relation.15.optional`, `relation.15.additional-properties-omitted`, `relation.15.root-any` (the sheet of the input values is the sheet of instances), `relation.15.one-sheet` (two molten places, one sheet of instances); `field.2.types` (additionalProperties), `relation.11.dependencies` (an optional member), `relation.4.all-of-narrows` (additionalProperties omitted in an allOf subschema), `relation.12.any-of-types` (a branch of one simple type, a property taken as not required) |
| relation.16 | `relation.16.once` (a member two branches name), `relation.16.covers` (a member a branch's pattern matches); `relation.11.any-of`, `relation.11.if-then-else`, `relation.11.dependencies`, `sheet.4.untitled-branch` |
| relation.17 | `relation.8.all-of` (no allOf column for a member named); `relation.11.if-then-else`, `relation.11.dependencies`, `sheet.4.untitled-branch` (a later sheet's column for a member written earlier, empty) |
| key.1       | every expected file                                                   |
| key.2       | every expected file's sheet of the input values                       |
| key.3       | `relation.13.array-of-arrays`, `relation.11.any-of` (parent, then pointer), `order.2.nested-order` (an instance's parent the instance that holds it), `relation.2.kind-holds-itself` (parent a row of the same sheet), `relation.12.object-branches` (a branch's parent the record that holds the location) |
| key.4       | `key.4.escaped-name` ('~0', '~1')                                     |
| key.5       | `relation.14.text` (a run keyed by its string's pointer)              |
| order.1     | `order.1.keywords`, `relation.2.kind-one-sheet` (a kind at its first reference), `order.1.shared-last` (the sheet of instances, then the sheet of runs, last), and the order of every case with more than one keyword |
| order.2     | `order.2.nested-order`, `value.1.order`                               |
| order.3     | `order.3.columns`; `relation.8.object` (a member in its property's place), `relation.8.all-of` (allOf's after the location's own) |
| file.1      | every case: one file, named by the caller                             |
| file.2      | `file.2.empty-sheet` (a sheet with no records left out), and every expected file |
| file.3      | every expected file                                                   |
| sheet.1     | `schema.2.title`                                                      |
| sheet.2     | `schema.11.ref` (a kind by its $ref's last token), `schema.11.ref-percent-encoded` (the token decoded), `relation.2.kind-one-sheet` |
| sheet.3     | `sheet.3.runs-name` (the sheet of runs, runs), `field.2.types` (the sheet of instances, instances) |
| sheet.4     | `sheet.4.untitled-branch` (no position for a keyword of one subschema), `sheet.4.several-patterns` (the pattern, where the keyword holds several); `relation.13.items-tuple` (positions), `relation.9.pattern-properties` (one pattern), `relation.13.array-of-arrays` (items), `relation.11.any-of` (titles), `relation.11.one-of` (positions), `relation.11.dependencies` (a key), `relation.8.object` (a property that relation.8 writes, in the name), `relation.12.any-of-in-items` (a branch named after the location it places) |
| sheet.5     | every expected file (parent, pointer, page, line, position); `relation.8.object` (property.member) |
| record.1    | every expected file                                                   |
| field.1     | `field.2.types`, `relation.15.optional`                               |
| field.2     | `field.2.types`, `relation.15.optional`                               |
| field.3     | `relation.14.text`, `relation.14.empty-runs` (an empty field), `sheet.3.runs-name` (a property's member) |
| field.4     | cannot-be-represented `field.4.lone-cr`, `field.4.surrogate`, `field.4.member-name` (HT, LF, FF, CR) |
| field.5     | cannot-be-represented `field.4.lone-cr`, `field.4.surrogate`, `field.4.member-name` |
| storage.1   | `storage.3.parts` (part 2 adds to part 1: its `all` holds both, and part 1's records unchanged) |
| storage.2   | `storage.2.two-inputs` (each input stored and communicated apart, named by its file); every case's `inputs` |
| storage.3   | `storage.3.parts` (every record, requested, is the input's MTSV file for the values stored); `storage.2.two-inputs`, `storage.4.rejected-part` |
| storage.4   | `storage.4.rejected-part` (part 2 rejected: nothing of it stored, value 2 included) |
| storage.5   | `storage.3.parts` (part 2's pointers begin at `/2`)                   |
| storage.6   | every case of `communicated/`: nothing is read but what is communicated |
| communication.1 | every expected file of `communicated/`, each an MTSV file         |
| communication.2 | every expected file of `communicated/`: values as stored, records in stored order |
| communication.3 | every `name.aggregate.mtsv`; `storage.2.two-inputs` (two inputs, in the order stored) |
| communication.4 | every `name.all.mtsv`; `storage.3.parts` request: values 1 to 3, the sheets at places 1 and 3 (a sheet holding none of them left out) |
| communication.5 | every `name.new-1.mtsv` (part 1) and `name.new-2.mtsv` (part 2 alone: part 1 was communicated at step 3) |
| communication.6 | every `name.again.mtsv`; every `name.new-2.mtsv`                  |

## Expected reports and rejections

Values are counted from 0, as key.2 counts positions, where JSON Lines
calls the first value 1; pointers are written as JSON strings (field.5,
value.10). A non-conforming input may break the rules of JSON Lines, as
`value.7.invalid-utf-8`, which is not UTF-8, and
`value.4.byte-order-mark` do.

| Case                          | Expected                                                   |
|-------------------------------|------------------------------------------------------------|
| `field.4.lone-cr`             | not carried: `"/0/t"`                                      |
| `field.4.surrogate`           | not carried: `"/0/t"`                                      |
| `field.4.member-name`         | not carried: `"/0/a\tb"`, `"/0/c\nd"`, `"/0/e\ff"`, `"/0/g\rh"` |
| `schema.1.schema-not-json`    | rejected: the module specification                        |
| `schema.1.not-a-schema`       | rejected: the module specification                        |
| `schema.2.no-title`           | rejected: the module specification                        |
| `schema.3.property-name`      | rejected: the module specification                        |
| `schema.3.title-name`         | rejected: the module specification                        |
| `schema.3.pattern-name`       | rejected: the module specification                        |
| `schema.3.dependency-name`    | rejected: the module specification                        |
| `schema.3.ref-name`           | rejected: the module specification                        |
| `schema.4.pattern-token`      | rejected: the module specification                        |
| `schema.5.ref-outside`        | rejected: the module specification                        |
| `schema.5.ref-leading-zero`   | rejected: the module specification                        |
| `schema.6.ref-plain-name`     | rejected: the module specification                        |
| `schema.6.ref-character`      | rejected: the module specification                        |
| `schema.7.ref-root`           | rejected: the module specification                        |
| `schema.8.ref-loop-all-of`    | rejected: the module specification                        |
| `schema.8.ref-loop-any-of`    | rejected: the module specification                        |
| `schema.8.ref-loop-one-of`    | rejected: the module specification                        |
| `schema.8.ref-loop-not`       | rejected: the module specification                        |
| `schema.8.ref-loop-if`        | rejected: the module specification                        |
| `schema.8.ref-loop-then`      | rejected: the module specification                        |
| `schema.8.ref-loop-else`      | rejected: the module specification                        |
| `schema.8.ref-loop-dependency` | rejected: the module specification                       |
| `value.4.byte-order-mark`     | rejected: value 1, pointer `""`                            |
| `value.7.not-json`            | rejected: value 1, pointer `""`                            |
| `value.7.invalid-utf-8`       | rejected: value 1, pointer `""`                            |
| `value.8.duplicate-names`     | rejected: value 0, pointer `""`                            |
| `value.9.schema-mismatch`     | rejected: value 1, pointer `"/n"`                          |
| `value.9.maximum`             | rejected: value 0, pointer `"/n"`                          |
| `value.10.deepest-place`      | rejected: value 0, pointer `"/p/k"`                        |
| `storage.4.rejected-part`     | part 2 rejected: value 3, pointer `"/n"`                   |
