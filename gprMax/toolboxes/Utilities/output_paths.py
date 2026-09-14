"""File-identity checks and transactional writes for processed outputs."""

from contextlib import contextmanager
import os
from pathlib import Path
import stat
import tempfile


def validate_output_path(
    filename, input_filenames, *, message="processed output must not overwrite an input HDF5 file"
):
    """Reject direct, symbolic-link and hard-link aliases of an input."""
    path = Path(filename)
    resolved_output = path.resolve()
    for input_filename in input_filenames:
        if not input_filename:
            continue
        input_path = Path(input_filename)
        aliases_input = resolved_output == input_path.resolve()
        if not aliases_input:
            try:
                aliases_input = path.samefile(input_path)
            except FileNotFoundError:
                aliases_input = False
        if aliases_input:
            raise ValueError(message)
    return path


@contextmanager
def atomic_output_path(filename):
    """Publish a completed sibling file, preserving the destination on error.

    The caller must close all handles to the temporary file before returning.
    Resolve a destination symlink to preserve the writer's existing target
    semantics. Replacement is on the same filesystem and never truncates an
    existing destination inode (including any of its other hard links).
    """
    destination = Path(filename).resolve()
    # A private staging directory lets the actual writer create the file with
    # its normal umask, unlike mkstemp's unconditional 0600 file permissions.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        yield temporary
        if destination.exists():
            temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
        os.replace(temporary, destination)
