"""The command's entry point."""

__all__ = ["main"]

from living_memory import _command


def main(argv: list[str] | None = None) -> None:
    """Convert a file or a provider's directory, or install a plugin."""
    _command.run(argv)


if __name__ == "__main__":
    main()
