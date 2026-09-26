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


class _Parser(argparse.ArgumentParser):
    # GNU Coding Standards 4.4: error messages from noninteractive
    # programs read "PROGRAM: MESSAGE" when there is no relevant source
    # file.
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {message}\n")


def main(argv: list[str] | None = None) -> None:
    """Convert a file or a directory to MTSV, or install into a host.

    argv -- the arguments, or None for those of the process

    Raise SystemExit with a GNU Coding Standards 4.4 message for a
    usage error, an input that does not conform, or a file that cannot
    be read or written.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Logging HOWTO, Configuring Logging for a Library: the
    # configuration of handlers is the prerogative of the application
    # developer.
    logging.basicConfig(format=f"{_PROG}: %(message)s", force=True)
    if args.install is not None:
        _install(args.install, parser)
        return
    if args.input is None or args.input == _STDIO:
        parser.error("an input file or directory is required to choose its reader")
    output = _output(args, parser)
    try:
        module = integrations.reader(args.input)
    except LookupError as error:
        parser.error(str(error))
    _convert(module, args.input, output)


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
    parser.add_argument("--install", metavar="HOST")
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


def _convert(module: ModuleType, path: Path, output: Path) -> None:
    # The integration supplies the schema and the values. Standard
    # output gets the whole file; an output file is kept as its
    # sheets' files, and a conversion appends to it only the values it
    # does not yet hold, one conversion at a time. An input that does
    # not conform is reported as "PROGRAM: MESSAGE" (GNU Coding
    # Standards 4.4).
    try:
        schema = module.schema()
        if output == _STDIO:
            text = convert(module.values(path, None), schema)
            sys.stdout.buffer.write(text.encode("utf-8"))
            return
        names = sheets(schema)
        with _store.locked(output):
            held = _store.repair(output, names)
            text = convert(module.values(path, held or None), schema, len(held))
            _store.append(output, text, names)
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {path}: {error}")
    except OSError as error:
        reason = error.strerror[:1].lower() + error.strerror[1:]
        raise SystemExit(f"{_PROG}: {error.filename}: {reason}")


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
    print(module.change())
    if input("Proceed? [y/N] ").strip().lower() in ("y", "yes"):
        module.install()
