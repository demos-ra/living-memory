"""How a person runs a conversion.

Functions:
run -- convert an OTLP JSON Lines file to an MTSV file
"""

__all__ = ["run"]

import argparse
import io
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import Any, NoReturn

import mtsv

from living_memory import integrations

# POSIX.1-2017 XBD 12.2, Guideline 13: the operand "-" means standard
# input, or standard output where an output file is meant.
_STDIO = Path("-")

# GNU Coding Standards 4.8.1: "The program's name should be a constant
# string".
_PROG = "living-memory"

# The draft, Media Type Registration: the file extension of MTSV.
_MTSV = ".mtsv"


class _Parser(argparse.ArgumentParser):
    """An argument parser whose errors read "PROGRAM: MESSAGE".

    GNU Coding Standards 4.4: error messages from noninteractive
    programs read "PROGRAM: MESSAGE" when there is no relevant source
    file.
    """

    def error(self, message: str) -> NoReturn:
        """Print the usage and "PROGRAM: MESSAGE", and exit with 2."""
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {message}\n")


def run(argv: list[str] | None = None) -> None:
    """Convert an OTLP JSON Lines file to an MTSV file.

    argv -- the arguments, or None for those of the process

    Raise SystemExit with a GNU Coding Standards 4.4 message for a
    usage error, a file that cannot be read or written, or a line that
    cannot be read.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = _output(args, parser)
    source = _format(args.input, parser)
    _convert(args.input, output, source)


def _build_parser() -> _Parser:
    """Return the parser of the command's arguments."""
    parser = _Parser(
        prog=_PROG,
        description="Convert an OTLP JSON Lines file to MTSV sheets.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("operand", type=Path, nargs="?", metavar="output")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--version", action="version", version=_notice())
    return parser


def _notice() -> str:
    """Return the version notice of GNU Coding Standards 4.8.1.

    The first line is the canonical name of the program, a space, and
    the version; then a copyright notice, the licence, that the program
    is free software, and that there is no warranty.
    """
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
    """Return the output operand, given or derived from the input.

    args -- the parsed arguments
    parser -- the parser that reports a usage error

    Exit with a usage error where the output is given twice, or where
    the input is standard input and no output is given.
    """
    if args.output is not None and args.operand is not None:
        parser.error("give the output file once, as an operand or with -o")
    output = args.operand if args.output is None else args.output
    if output is None:
        if args.input == _STDIO:
            parser.error("an output file is required to read standard input")
        output = args.input.with_suffix(_MTSV)
    return output


def _format(path: Path, parser: _Parser) -> str:
    """Return the file extension that names the input's format.

    path -- the input operand
    parser -- the parser that reports a usage error

    A stream is read as JSON Lines, the one input format. Exit with a
    usage error for an extension that names no format.
    """
    source = integrations.JSONL if path == _STDIO else path.suffix
    try:
        integrations.lookup(source)
    except LookupError as error:
        parser.error(str(error))
    return source


def _convert(path: Path, output: Path, source: str) -> None:
    """Convert the input file to the output file.

    path -- the input operand
    output -- the output operand
    source -- the input's file extension

    Raise SystemExit with a GNU Coding Standards 4.4 message for a file
    that cannot be read or written, or a line that cannot be read.
    """
    try:
        sheets = _read(source, path)
        _put(output, mtsv.dumps(sheets).encode("utf-8"))
    except ValueError as error:
        # GNU Coding Standards 4.4: "PROGRAM:SOURCEFILE:LINENO:
        # MESSAGE", and "PROGRAM: MESSAGE" when there is no relevant
        # source file.
        if path == _STDIO:
            raise SystemExit(f"{_PROG}: {error}")
        raise SystemExit(f"{_PROG}:{path}: {error}")
    except OSError as error:
        reason = error.strerror[:1].lower() + error.strerror[1:]
        raise SystemExit(f"{_PROG}: {error.filename}: {reason}")


def _read(source: str, path: Path) -> list[dict[str, Any]]:
    """Read sheets from a file, or from standard input, read whole.

    source -- the file extension that names the format
    path -- the input operand

    Raise OSError for a file that cannot be read, and ValueError for a
    line that cannot be read.
    """
    if path == _STDIO:
        data = sys.stdin.buffer.read()
    else:
        data = path.read_bytes()
    with io.BytesIO(data) as fp:
        return integrations.load(source, fp)


def _put(path: Path, data: bytes) -> None:
    """Write bytes to a file, or to standard output, in one call.

    path -- the output operand
    data -- the bytes

    Raise OSError for a file that cannot be written.
    """
    if path == _STDIO:
        sys.stdout.buffer.write(data)
    else:
        path.write_bytes(data)
