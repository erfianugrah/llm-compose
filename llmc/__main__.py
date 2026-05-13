"""Module entry point: `python -m llmc <subcommand>`.

Delegates to llmc.cli.main() so the CLI logic stays testable as a
regular function (without depending on sys.argv being set the right way).
"""

from llmc.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
