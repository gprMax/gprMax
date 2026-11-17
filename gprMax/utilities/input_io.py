"""UTF-8 text input shared by parsing, preflight and include expansion.

This module deliberately has no simulation/configuration dependencies:
study preflight reads input files before a SimulationConfig exists.
"""

from pathlib import Path


def read_input_text(filename):
    """Read a gprMax text input as UTF-8, tolerating a byte-order mark.

    gprMax input files are UTF-8. Reading with the process default encoding
    makes the result depend on the platform locale, and a leading byte-order
    mark would otherwise become part of the first command.

    Args:
        filename: path to the input file.

    Returns:
        text: complete file contents without the byte-order mark.

    Raises:
        ValueError: if the file is not valid UTF-8, naming the file.
    """

    try:
        return Path(filename).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as err:
        raise ValueError(
            f"Input file '{filename}' is not valid UTF-8. Re-save it as UTF-8; "
            "a byte-order mark is allowed."
        ) from err
