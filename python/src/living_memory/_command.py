"""The command: a source read by its integration, stored as MTSV in a
data bank, and communicated."""

__all__ = ["main"]

import argparse
import logging
import sys
from importlib.metadata import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import mtsv

from living_memory import _communication, _storage, _store, integrations
from living_memory._converter import convert, sheets

# POSIX.1-2017 XBD 12.2, Guideline 13: the operand "-" means standard
# input, or standard output where an output file is meant.
_STDIO = Path("-")
# GNU Coding Standards 4.8.1: "The program's name should be a constant
# string".
_PROG = "living-memory"
# The MTSV draft, Media Type Registration: the file extension of MTSV.
_MTSV = ".mtsv"

# The option that gives a host its context; a hook runs the command with
# it (install.mtsv › hooks.3).
_ADD_CONTEXT = "--add-context"
# A usage error's exit status, and a hook's: exit status 2 from a hook
# blocks, and from UserPromptSubmit erases the prompt (install.mtsv ›
# hooks.4).
_USAGE = 2
_FAILURE = 1
# A range of positions names its first and its last, both included, and
# a list of places is separated by semicolons (RFC 7111, 2.1. Row-Based
# Selection; 2.4. Multi-Selections; spec › communication.4).
_RANGE = "-"
_LIST = ";"


class _Parser(argparse.ArgumentParser):
    # GNU Coding Standards 4.4: error messages from noninteractive
    # programs read "PROGRAM: MESSAGE" when there is no relevant source
    # file.
    status = _USAGE

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(self.status, f"{self.prog}: {message}\n")


def main(argv: list[str] | None = None) -> None:
    """Store a source as MTSV, communicate what is stored, give a host
    its context, or install into a host.

    argv -- the arguments, or None for those of the process

    Raise SystemExit with a GNU Coding Standards 4.4 message for a
    usage error, an input that does not conform, or a file that cannot
    be read or written; with --add-context, a usage error exits with
    status 1, as every other failure does.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()
    if any(a == _ADD_CONTEXT or a.startswith(f"{_ADD_CONTEXT}=") for a in argv):
        parser.status = _FAILURE
    args = parser.parse_args(argv)
    # Logging HOWTO, Configuring Logging for a Library: the
    # configuration of handlers is the prerogative of the application
    # developer.
    logging.basicConfig(format=f"{_PROG}: %(message)s", force=True)
    options = ["--keep-files"] if args.keep_files else []
    if args.install is not None:
        _install(args.install, parser)
        return
    if args.input is None or args.input == _STDIO:
        parser.error("an input file or directory is required to choose its reader")
    output = _output(args, parser)
    try:
        module = (
            integrations.reader(args.input)
            if args.reader is None
            else integrations.named(args.reader)
        )
        host = (
            None
            if args.add_context is None
            else integrations.installer(args.add_context)
        )
    except LookupError as error:
        parser.error(str(error))
    request = _request(args, parser)
    if request is not None:
        _communicate(module, output, request)
        return
    _convert(module, args.input, output, options)
    if host is not None:
        _add_context(host, module, output)


def _build_parser() -> _Parser:
    # The options before the operands, single characters and GNU long
    # options (POSIX.1-2017 XBD 12.2; GNU Coding Standards 4.8); the
    # communications are named by the specification's verbs (spec ›
    # communication.3-5).
    parser = _Parser(
        prog=_PROG,
        description="Store a file or a directory as MTSV, by the integration"
        " that reads it, and communicate what is stored.",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("operand", type=Path, nargs="?", metavar="output")
    parser.add_argument("-o", "--output", type=Path)
    # Pandoc, Specifying formats: the input's format named with
    # -f/--from, else guessed from the input.
    parser.add_argument("-f", "--from", dest="reader", metavar="READER")
    # GNU Coding Standards 4.10: "keep-files", keeping the files a
    # program would otherwise remove.
    parser.add_argument("-k", "--keep-files", action="store_true")
    acts = parser.add_mutually_exclusive_group()
    acts.add_argument("--names", action="store_true")
    acts.add_argument("--filter", metavar="NAME")
    acts.add_argument("--new", metavar="NAME")
    acts.add_argument("--install", metavar="HOST")
    acts.add_argument(_ADD_CONTEXT, metavar="HOST")
    parser.add_argument("--values", metavar="FIRST-LAST")
    parser.add_argument("--places", metavar="PLACE;...")
    parser.add_argument("--version", action="version", version=_notice())
    return parser


def _epilog() -> str:
    # GNU Coding Standards 4.8.2: near the end, where to report bugs.
    return f"Report bugs to: {metadata('living-memory')['Author-email']}"


def _notice() -> str:
    # GNU Coding Standards 4.8.1: the program's canonical name and
    # version, then a copyright notice, the licence, that the program is
    # free software, and that there is no warranty.
    package = metadata("living-memory")
    return (
        f"{_PROG} {package['Version']}\n"
        "Copyright (C) 2026 Demos Ra\n"
        f"License {package['License-Expression']}\n"
        "This is free software: you are free to change and"
        " redistribute it.\n"
        "There is NO WARRANTY, to the extent permitted by law."
    )


def _output(args: argparse.Namespace, parser: _Parser) -> Path:
    # The output is given once, as an operand or with -o; else it is the
    # input's name, its extension replaced by .mtsv, beside it.
    if args.output is not None and args.operand is not None:
        parser.error("give the output file once, as an operand or with -o")
    output = args.operand if args.output is None else args.output
    return args.input.with_suffix(_MTSV) if output is None else output


def _request(args: argparse.Namespace, parser: _Parser) -> tuple | None:
    # A communication asked for: the names, a filter of one input's
    # records, or what is new of one; --values and --places narrow a
    # filter, --places what is new (spec › communication.3-5).
    places = _places(args.places, parser) if args.places is not None else None
    if args.values is not None and args.filter is None:
        parser.error("--values narrows --filter")
    if places is not None and args.filter is None and args.new is None:
        parser.error("--places narrows --filter or --new")
    if args.names:
        return ("names",)
    if args.filter is not None:
        values = _values(args.values, parser) if args.values is not None else None
        return ("filter", args.filter, values, places)
    if args.new is not None:
        return ("new", args.new, places)
    return None


def _values(text: str, parser: _Parser) -> tuple[int, int]:
    # A position, or a first and a last, both included, counted from 0.
    first, _, last = text.partition(_RANGE)
    if not first.isdigit() or (last and not last.isdigit()):
        parser.error(f"--values {text!r} is not a position or a range")
    return int(first), int(last or first)


def _places(text: str, parser: _Parser) -> list[int]:
    # Places separated by semicolons, each counted from 0.
    found = text.split(_LIST)
    if not all(place.isdigit() for place in found):
        parser.error(f"--places {text!r} is not a list of places")
    return [int(place) for place in found]


def _convert(module: ModuleType, path: Path, output: Path, options: list[str]) -> None:
    # The integration supplies the schema and the values, of its one
    # input, or of each input its lines give values to, each stored
    # apart by its name. Standard output gets the whole file; a data
    # bank stores only the values it does not yet hold, one conversion
    # at a time, and reads only the lines after those read; then, unless
    # kept, the files the integration names as spent are removed, and
    # how many lines are read is kept. An input that does not conform is
    # reported as "PROGRAM: MESSAGE" (GNU Coding Standards 4.4; spec ›
    # storage.1-5).
    try:
        schema = module.schema()
        by_line = hasattr(module, "lines")
        if output == _STDIO:
            index = module.lines(path, 0) if by_line else []
            found = (
                [
                    v
                    for n in module.inputs(index, [])
                    for v in module.values(path, n, None, index)
                ]
                if by_line
                else module.values(path, None)
            )
            text = convert(found, schema)
            sys.stdout.buffer.write(text.encode("utf-8"))
            return
        sheet_names = sheets(schema)
        with _store.locked(output):
            if not by_line:
                _insert(module, (path, path.stem, []), output, (schema, sheet_names))
                return
            start = _store.read(output)
            index = module.lines(path, start)
            held: list[dict[str, str]] = []
            for name in module.inputs(index, _store.inputs(output)):
                held += _insert(
                    module, (path, name, index), output, (schema, sheet_names)
                )
                if "--keep-files" not in options and hasattr(module, "spent"):
                    now = _held(output / name, sheet_names)
                    for spent in module.spent(path, name, now, index):
                        spent.unlink(missing_ok=True)
            _store.mark_read(output, module.read(index, held, start))
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {path}: {error}")
    except OSError as error:
        reason = error.strerror[:1].lower() + error.strerror[1:]
        raise SystemExit(f"{_PROG}: {error.filename}: {reason}")


def _insert(
    module: ModuleType,
    source: tuple[Path, str, list],
    output: Path,
    schema: tuple[bytes, list[str]],
) -> list[dict[str, str]]:
    # One input's next part, from the number of its values stored,
    # inserted whole or not at all; the records of its sheet of the
    # input values are returned as it now holds them (spec ›
    # storage.1-5).
    path, name, index = source
    folder = output / name
    text_schema, sheet_names = schema
    _store.repair(folder, sheet_names)
    held = _held(folder, sheet_names)
    found = (
        module.values(path, name, held or None, index)
        if hasattr(module, "lines")
        else module.values(path, held or None)
    )
    start = _storage.count(_store.stored(folder, sheet_names))
    text = convert(found, text_schema, start)
    _store.append(folder, text, sheet_names)
    now = _held(folder, sheet_names)
    if now:
        _store.add_input(output, name)
    return now


def _held(folder: Path, sheet_names: list[str]) -> list[dict[str, str]]:
    # The records of an input's sheet of the input values, as the data
    # bank communicates them, each by its header's names (spec ›
    # communication.4).
    stored = _store.stored(folder, sheet_names)
    values = _storage.count(stored)
    text = _communication.records(stored, 0, values - 1, [0])
    return [
        dict(zip(sheet["header"], record))
        for sheet in mtsv.loads(text)
        for record in sheet["records"]
    ]


def _communicate(module: ModuleType, output: Path, request: tuple) -> None:
    # A communication, as MTSV on standard output; what is new advances
    # the number communicated (spec › communication.1-6).
    sheet_names = sheets(module.schema())
    with _store.locked(output):
        bank = _Bank(output, sheet_names)
        if request[0] == "names":
            text = bank.names()
        else:
            if request[1] not in _store.inputs(output):
                raise SystemExit(f"{_PROG}: {output}: holds no input {request[1]!r}")
            if request[0] == "filter":
                text = bank.filter(request[1], request[2], request[3])
            else:
                text = bank.new(request[1], request[2])
    sys.stdout.buffer.write(text.encode("utf-8"))


class _Bank:
    # What a host is given of a data bank: only its communications (spec
    # › storage.6, communication.1-6).
    def __init__(self, output: Path, sheet_names: list[str]) -> None:
        self.output = output
        self.sheet_names = sheet_names

    def inputs(self) -> list[str]:
        return _store.inputs(self.output)

    def names(self) -> str:
        return _communication.names(
            [(name, self._stored(name)) for name in self.inputs()]
        )

    def filter(
        self, name: str, values: tuple[int, int] | None, places: list[int] | None
    ) -> str:
        stored = self._stored(name)
        first, last = values if values is not None else (0, _storage.count(stored) - 1)
        return _communication.records(stored, first, last, places)

    def new(self, name: str, places: list[int] | None) -> str:
        # What is new, from the number communicated to the last value
        # stored, the number then kept (spec › communication.5,
        # communication.6).
        folder = self.output / name
        stored = self._stored(name)
        values = _storage.count(stored)
        text = _communication.records(
            stored, _store.communicated(folder), values - 1, places
        )
        _store.mark_communicated(folder, values)
        return text

    def _stored(self, name: str) -> Any:
        return _store.stored(self.output / name, self.sheet_names)


def _add_context(host: ModuleType, module: ModuleType, output: Path) -> None:
    # The host's context: it reads the hook's input on standard input
    # and is given the data bank's communications; its text goes to
    # standard output.
    try:
        with _store.locked(output):
            bank = _Bank(output, sheets(module.schema()))
            text = host.context(sys.stdin.buffer.read(), bank)
    except ValueError as error:
        raise SystemExit(f"{_PROG}: --add-context: {error}")
    sys.stdout.buffer.write(text.encode("utf-8"))


def _install(host: str, parser: _Parser) -> None:
    # Before changing another program's configuration, the command
    # states the change and asks, No by default, and refuses without a
    # terminal (GNU Coding Standards 4.10, --interactive).
    try:
        module = integrations.installer(host)
    except LookupError as error:
        parser.error(str(error))
    if not sys.stdin.isatty():
        raise SystemExit(f"{_PROG}: --install={host} asks first, and needs a terminal")
    try:
        print(module.change())
        keep = _yes(module.question())
        if _yes("Proceed?"):
            module.install(["--keep-files"] if keep else [])
    except OSError as error:
        raise SystemExit(f"{_PROG}: --install={host}: {error}")


def _yes(question: str) -> bool:
    # A question answered at the terminal, No by default.
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
