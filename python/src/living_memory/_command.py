"""How a person runs a conversion."""

__all__ = ["run"]

import argparse
import io
import logging
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import Any, NoReturn

import mtsv

from living_memory import OTLPDecodeError, integrations

# POSIX.1-2017 XBD 12.2, Guideline 13.
_STDIO = Path("-")

# GNU Coding Standards 4.8.1.
_PROG = "living-memory"

# The draft, Media Type Registration.
_MTSV = ".mtsv"


# GNU Coding Standards 4.4.
class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {message}\n")


def run(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = _output(args, parser)
    source = _format(args.input, parser)
    # Logging HOWTO, Configuring Logging for a Library.
    logging.basicConfig(format=f"{_PROG}: %(message)s", force=True)
    _convert(args.input, output, source)


def _build_parser() -> _Parser:
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
    # GNU Coding Standards 4.8.1.
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
    if args.output is not None and args.operand is not None:
        parser.error("give the output file once, as an operand or with -o")
    output = args.operand if args.output is None else args.output
    if output is None:
        if args.input == _STDIO:
            parser.error("an output file is required to read standard input")
        output = args.input.with_suffix(_MTSV)
    return output


def _format(path: Path, parser: _Parser) -> str:
    source = integrations.JSONL if path == _STDIO else path.suffix
    try:
        integrations.lookup(source)
    except LookupError as error:
        parser.error(str(error))
    return source


def _convert(path: Path, output: Path, source: str) -> None:
    # GNU Coding Standards 4.4.
    try:
        sheets = _read(source, path)
        _put(output, mtsv.dumps(sheets).encode("utf-8"))
    except OTLPDecodeError as error:
        if path == _STDIO:
            raise SystemExit(f"{_PROG}: {error}")
        raise SystemExit(f"{_PROG}:{path}:{error.lineno}: {error.msg}")
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {error}")
    except OSError as error:
        reason = error.strerror[:1].lower() + error.strerror[1:]
        raise SystemExit(f"{_PROG}: {error.filename}: {reason}")


def _read(source: str, path: Path) -> list[dict[str, Any]]:
    if path == _STDIO:
        data = sys.stdin.buffer.read()
    else:
        data = path.read_bytes()
    with io.BytesIO(data) as fp:
        return integrations.load(source, fp)


def _put(path: Path, data: bytes) -> None:
    if path == _STDIO:
        sys.stdout.buffer.write(data)
    else:
        path.write_bytes(data)
