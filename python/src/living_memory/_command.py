"""The command: a source read by its integration, written as MTSV."""

__all__ = ["main"]

import argparse
import logging
import sys
from importlib.metadata import metadata
from pathlib import Path
from types import ModuleType
from typing import NoReturn

from living_memory import _store, integrations
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


class _Parser(argparse.ArgumentParser):
    # GNU Coding Standards 4.4: error messages from noninteractive
    # programs read "PROGRAM: MESSAGE" when there is no relevant source
    # file.
    status = _USAGE

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(self.status, f"{self.prog}: {message}\n")


def main(argv: list[str] | None = None) -> None:
    """Convert a file or a directory to MTSV, give a host its context of
    the output, or install into a host.

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
    _convert(module, args.input, output, options)
    if host is not None:
        _add_context(host, module, args.input, output)


def _build_parser() -> _Parser:
    # The options before the operands, single characters and GNU long
    # options (POSIX.1-2017 XBD 12.2; GNU Coding Standards 4.8).
    parser = _Parser(
        prog=_PROG,
        description="Convert a file or a directory to MTSV, by the integration"
        " that reads it.",
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
    parser.add_argument("--install", metavar="HOST")
    parser.add_argument(_ADD_CONTEXT, metavar="HOST")
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


def _convert(module: ModuleType, path: Path, output: Path, options: list[str]) -> None:
    # The integration supplies the schema and the values, of its one
    # input, or of each input its lines give values to, each written to
    # a folder of the output by its name. Standard output gets the whole
    # file; an output file is kept as its sheets' files, and a
    # conversion reads only the lines after those read, and appends only
    # the values the output does not yet hold, one conversion at a time;
    # then, unless kept, the files the integration names as spent are
    # removed, and how many lines are read is kept. An input that does
    # not conform is reported as "PROGRAM: MESSAGE" (GNU Coding
    # Standards 4.4).
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
                _append(
                    module, (path, None, []), output, (schema, sheet_names), options
                )
                return
            start = _store.read(output)
            index = module.lines(path, start)
            held: list[dict[str, str]] = []
            for name in module.inputs(index, _store.inputs(output)):
                held += _append(
                    module, (path, name, index), output, (schema, sheet_names), options
                )
            _store.mark_read(output, module.read(index, held, start))
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {path}: {error}")
    except OSError as error:
        reason = error.strerror[:1].lower() + error.strerror[1:]
        raise SystemExit(f"{_PROG}: {error.filename}: {reason}")


def _append(
    module: ModuleType,
    source: tuple[Path, str | None, list],
    output: Path,
    schema: tuple[bytes, list[str]],
    options: list[str],
) -> list[dict[str, str]]:
    # One input's new values appended to its output, then its spent
    # files removed unless they are kept; the records of its sheet of
    # the input values are returned as it now holds them.
    path, name, index = source
    folder = output if name is None else output / name
    text_schema, sheet_names = schema
    _store.repair(folder, sheet_names)
    held = _store.held(folder, sheet_names)
    found = (
        module.values(path, held or None)
        if name is None
        else module.values(path, name, held or None, index)
    )
    text = convert(found, text_schema, len(held))
    _store.append(folder, text, sheet_names)
    now = _store.held(folder, sheet_names)
    if name is not None and "--keep-files" not in options and hasattr(module, "spent"):
        for spent in module.spent(path, name, now, index):
            spent.unlink(missing_ok=True)
    return now


def _add_context(
    host: ModuleType, module: ModuleType, path: Path, output: Path
) -> None:
    # The host's context: it reads the hook's input on standard input
    # and the output as the store gives it, the inputs being those the
    # output holds, and says what each input's output has now given; its
    # text goes to standard output.
    names = _store.inputs(output)
    sheet_names = sheets(module.schema())

    def look(name: str) -> dict:
        # An input's output, each part read only when asked for.
        folder = output / name
        return {
            "sheets": lambda: _store.view(folder),
            "held": lambda: _store.held(folder, sheet_names),
            "given": lambda: _store.given(folder),
        }

    try:
        with _store.locked(output):
            text, marks = host.context(sys.stdin.buffer.read(), output, names, look)
            for name, count in marks.items():
                _store.mark_given(output / name, count)
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
