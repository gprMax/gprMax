"""List and copy the version-matched examples distributed with gprMax."""

from __future__ import annotations

import argparse
import shutil
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional, Sequence

from gprMax._version import __version__


EXAMPLES_PACKAGE = "gprMax._examples"


def _is_example_file(path: Path) -> bool:
    """Return whether a path is a distributable example resource."""

    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != "__init__.py"
    )


@contextmanager
def _example_source() -> Iterator[Path]:
    """Yield the installed example resource directory.

    Editable source trees created before the resource package was installed
    use the repository's top-level examples directory as a fallback.
    """

    try:
        files = getattr(resources, "files", None)
        if files is not None:
            root = files(EXAMPLES_PACKAGE)
            if isinstance(root, Path):
                yield root
                return

            # Wheels are installed unpacked because gprMax is not zip safe.
            # This fallback also supports other importers exposing Traversable
            # resources without assuming that their directory is a real path.
            with resources.as_file(root.joinpath("README.rst")) as readme:
                yield readme.parent
                return

        with resources.path(EXAMPLES_PACKAGE, "README.rst") as readme:
            yield readme.parent
            return
    except (ModuleNotFoundError, FileNotFoundError):
        source_tree = Path(__file__).resolve().parents[1] / "examples"
        if not source_tree.is_dir():
            raise RuntimeError("The installed gprMax examples could not be located") from None
        yield source_tree


def default_destination() -> Path:
    """Return the default writable workspace for copied examples."""

    return Path.cwd() / f"gprMax-examples-{__version__}"


def copy_examples(destination: Optional[Path] = None, *, force: bool = False) -> Path:
    """Copy installed examples into ``destination/examples``.

    Parameters
    ----------
    destination
        Workspace directory. A versioned directory in the current working
        directory is used when omitted.
    force
        Permit files in an existing ``examples`` directory to be overwritten.
        Unrelated files are retained.
    """

    workspace = (Path(destination) if destination is not None else default_destination()).expanduser().resolve()
    target = workspace / "examples"

    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; choose another destination or pass --force to update it"
        )

    workspace.mkdir(parents=True, exist_ok=True)
    with _example_source() as source:
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=force,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    return workspace


def list_examples() -> list[tuple[str, int]]:
    """Return top-level example categories and their model/resource counts."""

    with _example_source() as source:
        return [
            (directory.name, sum(1 for path in directory.rglob("*") if _is_example_file(path)))
            for directory in sorted(source.iterdir())
            if directory.is_dir() and directory.name != "__pycache__"
        ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gprMax.examples",
        description="List or copy the examples matching this gprMax installation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list the installed example categories")

    copy_parser = subparsers.add_parser("copy", help="copy examples to a writable workspace")
    copy_parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help=f"workspace to create (default: ./gprMax-examples-{__version__})",
    )
    copy_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite matching files if the destination already contains examples",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""

    args = _build_parser().parse_args(argv)

    if args.command == "list":
        print(f"Examples distributed with gprMax {__version__}:")
        for category, count in list_examples():
            print(f"  {category:<20} {count:>3} files")
        return 0

    try:
        workspace = copy_examples(args.destination, force=args.force)
    except FileExistsError as error:
        raise SystemExit(str(error)) from None

    print(f"Copied gprMax {__version__} examples to {workspace / 'examples'}")
    print(f"Run them from {workspace}, for example:")
    print("  python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
