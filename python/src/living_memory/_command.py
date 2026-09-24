"""How a person runs a conversion, or installs a plugin."""

__all__ = ["run"]

import argparse
import io
import logging
import os
import subprocess
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import Any, NoReturn

import mtsv

from living_memory import _otlp_json, providers

# The operand "-" is standard input, or standard output where an output
# is meant (POSIX.1-2017 XBD 12.2, Guideline 13).
_STDIO = Path("-")

# The program's name is a constant string (GNU Coding Standards 4.8.1).
_PROG = "living-memory"

# A file of this extension is OTLP JSON Lines, and an output takes
# MTSV's extension (JSONL, Conventions; MTSV-DRAFT, Media Type
# Registration).
_JSONL = ".jsonl"
_MTSV = ".mtsv"

# Only y acts on the question (install.mtsv › prompt.1).
_YES = "y"


class _Parser(argparse.ArgumentParser):
    # An error reads "PROGRAM: MESSAGE" (GNU Coding Standards 4.4).
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {message}\n")


def run(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.install is not None:
        _install(args, parser)
        return
    if args.input is None:
        parser.error("an input file or directory is required")
    output = _output(args, parser)
    _check_input(args.input, parser)
    # The application, not the library, configures the handler
    # (Logging HOWTO, Configuring Logging for a Library).
    logging.basicConfig(format=f"{_PROG}: %(message)s", force=True)
    _convert(args.input, output)


def _build_parser() -> _Parser:
    parser = _Parser(
        prog=_PROG,
        description=(
            "Convert an OTLP JSON Lines file, or a provider's directory,"
            " to MTSV sheets; or install a plugin."
        ),
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("operand", type=Path, nargs="?", metavar="output")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--install", metavar="PLUGIN")
    parser.add_argument("--version", action="version", version=_notice())
    return parser


def _notice() -> str:
    # The name and version, then the copyright, the licence, that it is
    # free software, and that there is no warranty (GNU Coding Standards
    # 4.8.1).
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
        output = _derived(args.input)
    return output


def _derived(path: Path) -> Path:
    # The output is named after the input, beside it: a file's extension
    # is replaced by MTSV's, and a directory, having none, takes it
    # after its whole name.
    if _is_directory(path):
        return path.with_name(path.name + _MTSV)
    return path.with_suffix(_MTSV)


def _check_input(path: Path, parser: _Parser) -> None:
    # A file is read by its extension, a directory by the provider whose
    # directory it is, and a stream as OTLP JSON Lines (PANDOC,
    # Specifying formats).
    if _is_directory(path):
        try:
            providers.lookup(path)
        except LookupError as error:
            parser.error(str(error))
    elif path != _STDIO and path.suffix != _JSONL:
        parser.error(f"no format for {path.suffix!r}")


def _is_directory(path: Path) -> bool:
    return path != _STDIO and path.is_dir()


def _convert(path: Path, output: Path) -> None:
    # A file's error names the file and line, a stream's the line only
    # (GNU Coding Standards 4.4).
    try:
        sheets = _read(path)
        _put(output, mtsv.dumps(sheets).encode("utf-8"))
    except _otlp_json.OTLPDecodeError as error:
        if path == _STDIO:
            raise SystemExit(f"{_PROG}: {error}")
        raise SystemExit(f"{_PROG}:{path}:{error.lineno}: {error.msg}")
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {error}")
    except OSError as error:
        raise SystemExit(_os_error(error))


def _read(path: Path) -> list[dict[str, Any]]:
    if _is_directory(path):
        return providers.lookup(path).load(path)
    if path == _STDIO:
        data = sys.stdin.buffer.read()
    else:
        data = path.read_bytes()
    with io.BytesIO(data) as fp:
        return _otlp_json.load(fp)


def _install(args: argparse.Namespace, parser: _Parser) -> None:
    # The plugin's plan is stated, asked about, then carried out step by
    # step, stopping at the first that fails (install.mtsv ›
    # conformance.2, plugin.1).
    if args.input is not None or args.operand is not None or args.output:
        parser.error("--install takes no input or output")
    try:
        plugin = providers.lookup_plugin(args.install)
        changes, steps = plugin.plan(os.environ, sys.platform, Path.home())
    except LookupError as error:
        parser.error(str(error))
    if not _proceed(changes):
        return
    try:
        for step in steps:
            _carry_out(*step)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{_PROG}: {' '.join(error.cmd)}: exit {error.returncode}")
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {error}")
    except OSError as error:
        raise SystemExit(_os_error(error))


def _proceed(changes: str) -> bool:
    # No is the default and only y acts; with no terminal, nothing is
    # asked and nothing changes (install.mtsv › prompt.1).
    if not sys.stdin.isatty():
        raise SystemExit(f"{_PROG}: --install asks first, and needs a terminal")
    print(changes)
    return input("Proceed? [y/N] ").strip() == _YES


def _carry_out(action: str, target: Any, detail: Any) -> None:
    if action == "directory":
        mode = 0o777 if detail is None else detail
        target.mkdir(mode=mode, parents=True, exist_ok=True)
    elif action == "file":
        target.touch(exist_ok=True)
    elif action == "run":
        subprocess.run(target, check=True)
    elif action == "replace":
        document = target.read_text("utf-8") if target.exists() else None
        _put(target, detail(document).encode("utf-8"))
    else:
        raise ValueError(f"no such step {action!r}")


def _put(path: Path, data: bytes) -> None:
    # A file is replaced whole: written beside its name in the same
    # folder and renamed onto it (POSIX.1-2017 XSH rename).
    if path == _STDIO:
        sys.stdout.buffer.write(data)
        return
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    with part.open("xb") as file:
        file.write(data)
    if path.exists():
        os.chmod(part, path.stat().st_mode)
    os.replace(part, path)


def _os_error(error: OSError) -> str:
    # A system error names the file, its message not capitalized
    # (GNU Coding Standards 4.4).
    reason = error.strerror[:1].lower() + error.strerror[1:]
    return f"{_PROG}: {error.filename}: {reason}"
