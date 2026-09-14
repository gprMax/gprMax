"""Fail-closed checks for the release workflow (never imports the solver)."""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


PYTHONS = ("cp311", "cp312", "cp313")
PLATFORMS = ("linux-x86_64", "windows-amd64", "macos-x86_64", "macos-arm64")
EXPECTED_WHEELS = {(python, platform) for python in PYTHONS for platform in PLATFORMS}


def release_version(path: Path, ref: str, destination: str) -> str:
    """Read a literal version and require its exact tag for either index."""
    if destination not in {"dry-run", "testpypi", "pypi"}:
        raise ValueError(f"Unknown destination: {destination}")
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one literal __version__ string")
    value = values[0]
    version = Version(value)
    if str(version) != value or version.local is not None or version.epoch:
        raise ValueError(f"Use a canonical public version without a local suffix or epoch: {value}")
    if destination != "dry-run" and ref != f"refs/tags/v{value}":
        raise ValueError(f"Publishing requires refs/tags/v{value}; selected {ref}")
    return value


def _identity(name: str, version: Version | str, expected: str) -> None:
    if canonicalize_name(name) != "gprmax" or Version(str(version)) != Version(expected):
        raise ValueError(f"Expected gprMax {expected}, found {name} {version}")


def _metadata(data: bytes, version: str) -> None:
    metadata = BytesParser().parsebytes(data)
    if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
        raise ValueError("Distribution metadata must contain one Name and Version")
    _identity(metadata["Name"], metadata["Version"], version)


def _platform(tag: str) -> str:
    legacy_manylinux = {"manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"}
    if (tag.startswith("manylinux_") and tag.endswith("_x86_64")) or tag in legacy_manylinux:
        return "linux-x86_64"
    if tag == "win_amd64":
        return "windows-amd64"
    for arch in ("x86_64", "arm64"):
        if tag.startswith("macosx_") and tag.endswith(f"_{arch}"):
            return f"macos-{arch}"
    raise ValueError(f"Unexpected or non-portable wheel platform: {tag}")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_distributions(directory: Path, version: str) -> dict:
    """Require exactly one source archive and the complete tested wheel matrix."""
    seen = set()
    sdists = 0
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected release entry: {path.name}")
        if path.name.endswith(".whl"):
            name, found_version, build, tags = parse_wheel_filename(path.name)
            _identity(name, found_version, version)
            if build:
                raise ValueError(f"Unexpected wheel build suffix: {path.name}")
            keys = set()
            for tag in tags:
                if tag.interpreter not in PYTHONS or tag.abi != tag.interpreter:
                    raise ValueError(f"Unexpected Python/ABI tag: {tag}")
                keys.add((tag.interpreter, _platform(tag.platform)))
            if len(keys) != 1 or seen.intersection(keys):
                raise ValueError(f"Duplicate or ambiguous wheel matrix entry: {path.name}")
            seen.update(keys)
            with zipfile.ZipFile(path) as archive:
                metadata_paths = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
                if len(metadata_paths) != 1:
                    raise ValueError(f"Expected one METADATA file: {path.name}")
                _metadata(archive.read(metadata_paths[0]), version)
                wheel_path = metadata_paths[0].removesuffix("METADATA") + "WHEEL"
                wheel = BytesParser().parsebytes(archive.read(wheel_path))
                metadata_tags = set()
                for value in wheel.get_all("Tag", []):
                    metadata_tags.update(parse_tag(value))
                if metadata_tags != tags:
                    raise ValueError(f"Filename/WHEEL tags disagree: {path.name}")
        elif path.name.endswith(".tar.gz"):
            name, found_version = parse_sdist_filename(path.name)
            _identity(name, found_version, version)
            sdists += 1
            # Read metadata without extracting or executing any archive contents.
            with tarfile.open(path, "r:gz") as archive:
                metadata = archive.extractfile(path.name.removesuffix(".tar.gz") + "/PKG-INFO")
                if metadata is None:
                    raise ValueError(f"Missing PKG-INFO: {path.name}")
                _metadata(metadata.read(), version)
        else:
            raise ValueError(f"Unexpected release file: {path.name}")
        files[path.name] = {"sha256": sha256(path), "size": path.stat().st_size}
    if sdists != 1 or seen != EXPECTED_WHEELS:
        raise ValueError(
            f"Incomplete release: {sdists} source archives; missing wheels {sorted(EXPECTED_WHEELS - seen)}"
        )
    return {"version": version, "files": files}


def check_downloaded(directory: Path, manifest: dict) -> None:
    """Check a wheel selected from artifacts or an index against its build hash."""
    paths = list(directory.iterdir())
    if len(paths) != 1 or paths[0].suffix != ".whl" or paths[0].is_symlink():
        raise ValueError("Expected exactly one downloaded wheel")
    path = paths[0]
    expected = manifest["files"].get(path.name)
    if expected is None or expected != {"sha256": sha256(path), "size": path.stat().st_size}:
        raise ValueError(f"Downloaded wheel differs from the validated release: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument("--version-file", type=Path, default=Path("gprMax/_version.py"))
    version.add_argument("--ref", required=True)
    version.add_argument("--destination", choices=("dry-run", "testpypi", "pypi"), required=True)
    distributions = commands.add_parser("distributions")
    distributions.add_argument("directory", type=Path)
    distributions.add_argument("--version", required=True)
    distributions.add_argument("--manifest", type=Path, required=True)
    downloaded = commands.add_parser("downloaded")
    downloaded.add_argument("directory", type=Path)
    downloaded.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "version":
        print(release_version(args.version_file, args.ref, args.destination))
    elif args.command == "distributions":
        manifest = check_distributions(args.directory, args.version)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Validated {len(manifest['files'])} distributions for {args.version}")
    else:
        check_downloaded(args.directory, json.loads(args.manifest.read_text()))
        print("Downloaded wheel matches the validated release")


if __name__ == "__main__":
    main()
