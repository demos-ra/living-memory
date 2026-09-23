"""The command's entry point.

Functions:
main -- convert an OTLP JSON Lines file to an MTSV file
"""

__all__ = ["main"]

from living_memory import _command


def main(argv: list[str] | None = None) -> None:
    """Convert an OTLP JSON Lines file to an MTSV file.

    argv -- the arguments, or None for those of the process
    """
    _command.run(argv)


if __name__ == "__main__":
    main()
