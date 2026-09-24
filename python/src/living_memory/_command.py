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

from living_memory import OTLPDecodeError, integrations

# POSIX.1-2017 XBD 12.2, Guideline 13.
_STDIO = Path("-")

# GNU Coding Standards 4.8.1.
_PROG = "living-memory"

# The draft, Media Type Registration.
_MTSV = ".mtsv"

# install.mtsv › prompt.1.
_YES = "y"
# install.mtsv › directory.2; XDG Base Directory, Referencing this
# specification, block 9.
_PRIVATE = 0o700


# GNU Coding Standards 4.4.
class _Parser(argparse.ArgumentParser):
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
    source = _format(args.input, parser)
    # Logging HOWTO, Configuring Logging for a Library.
    logging.basicConfig(format=f"{_PROG}: %(message)s", force=True)
    _convert(args.input, output, source)


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
        if _is_directory(path):
            integrations.lookup_directory(path)
        else:
            integrations.lookup(source)
    except LookupError as error:
        parser.error(str(error))
    return source


def _is_directory(path: Path) -> bool:
    # A file is read by its extension, a directory by its provider.
    return path != _STDIO and path.is_dir()


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
        raise SystemExit(_os_error(error))


def _read(source: str, path: Path) -> list[dict[str, Any]]:
    if _is_directory(path):
        return integrations.lookup_directory(path).load(path)
    if path == _STDIO:
        data = sys.stdin.buffer.read()
    else:
        data = path.read_bytes()
    with io.BytesIO(data) as fp:
        return integrations.load(source, fp)


def _install(args: argparse.Namespace, parser: _Parser) -> None:
    # install.mtsv › conformance.2: prompt, directory, plugin,
    # capture.
    if args.input is not None or args.operand is not None or args.output:
        parser.error("--install takes no input or output")
    try:
        plugin = integrations.lookup_plugin(args.install)
        data = plugin.data_directory(os.environ, sys.platform, Path.home())
    except LookupError as error:
        parser.error(str(error))
    if not _proceed(plugin.changes(data, Path.home())):
        return
    try:
        _prepare(plugin.directories(data), plugin.index_file(data))
        for command in plugin.commands(data):
            subprocess.run(command, check=True)
        settings = Path.home() / plugin.SETTINGS
        settings.parent.mkdir(parents=True, exist_ok=True)
        document = settings.read_text("utf-8") if settings.exists() else None
        _put(settings, plugin.settings(document, data).encode("utf-8"))
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{_PROG}: {' '.join(error.cmd)}: exit {error.returncode}")
    except ValueError as error:
        raise SystemExit(f"{_PROG}: {error}")
    except OSError as error:
        raise SystemExit(_os_error(error))


def _proceed(changes: str) -> bool:
    # install.mtsv › prompt.1: No by default, only y acts; no terminal,
    # no question.
    if not sys.stdin.isatty():
        raise SystemExit(f"{_PROG}: --install asks first, and needs a terminal")
    print(changes)
    return input("Proceed? [y/N] ").strip() == _YES


def _prepare(directories: list[Path], index: Path) -> None:
    # install.mtsv › directory.2: the directories private, an empty
    # index file.
    for directory in directories:
        directory.mkdir(mode=_PRIVATE, parents=True, exist_ok=True)
    index.touch(exist_ok=True)


def _put(path: Path, data: bytes) -> None:
    # POSIX.1-2017 XSH rename: the file replaced whole, written beside
    # its name in the same folder and renamed onto it.
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
    # GNU Coding Standards 4.4.
    reason = error.strerror[:1].lower() + error.strerror[1:]
    return f"{_PROG}: {error.filename}: {reason}"
