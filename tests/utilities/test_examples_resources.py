"""Tests for installed, version-matched example resources."""

from pathlib import Path

import pytest

from gprMax import examples
from gprMax._version import __version__
from packaging_config import EXCLUDED_PACKAGE_DATA, distribution_packages


@pytest.mark.unit
def test_example_categories_are_available():
    categories = dict(examples.list_examples())

    assert {"gpr", "antennas", "features", "jupyter-notebooks", "rcs"} <= categories.keys()
    assert all(count > 0 for count in categories.values())


@pytest.mark.unit
def test_copy_examples_uses_versioned_default_and_preserves_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    workspace = examples.copy_examples()

    assert workspace == tmp_path / f"gprMax-examples-{__version__}"
    assert (workspace / "examples" / "README.rst").is_file()
    assert (workspace / "examples" / "gpr" / "basic" / "cylinder_Ascan_2D.in").is_file()
    assert not list((workspace / "examples").rglob("__pycache__"))
    assert not list((workspace / "examples").rglob("*.pyc"))


@pytest.mark.unit
def test_copy_examples_refuses_existing_tree_without_force(tmp_path):
    destination = tmp_path / "workspace"
    examples.copy_examples(destination)

    with pytest.raises(FileExistsError, match="already exists"):
        examples.copy_examples(destination)


@pytest.mark.unit
def test_copy_examples_force_updates_files_and_preserves_unrelated_files(tmp_path):
    destination = tmp_path / "workspace"
    examples.copy_examples(destination)
    marker = destination / "examples" / "local-notes.txt"
    marker.write_text("keep", encoding="utf-8")
    readme = destination / "examples" / "README.rst"
    readme.write_text("stale", encoding="utf-8")

    examples.copy_examples(destination, force=True)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert readme.read_text(encoding="utf-8").startswith("==============")


@pytest.mark.unit
def test_examples_cli_lists_and_copies(tmp_path, capsys):
    assert examples.main(["list"]) == 0
    assert "Examples distributed with gprMax" in capsys.readouterr().out

    destination = tmp_path / "copied"
    assert examples.main(["copy", str(destination)]) == 0
    output = capsys.readouterr().out
    assert str(destination / "examples") in output
    assert "python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in" in output


@pytest.mark.unit
def test_wheel_packages_exclude_developer_archives_but_keep_toolboxes_and_examples():
    packages = set(distribution_packages())

    assert "gprMax" in packages
    assert "gprMax._examples" in packages
    assert "toolboxes" in packages
    assert not any(name == "testing" or name.startswith("testing.") for name in packages)
    assert not any(name == "reframe_tests" or name.startswith("reframe_tests.") for name in packages)


@pytest.mark.unit
def test_large_generated_toolbox_assets_are_excluded_from_wheels():
    cython_exclusions = EXCLUDED_PACKAGE_DATA["gprMax.cython"]
    step_exclusions = EXCLUDED_PACKAGE_DATA["toolboxes.STEPtoVoxel"]
    stl_exclusions = EXCLUDED_PACKAGE_DATA["toolboxes.STLtoVoxel"]

    assert {"*.c", "*.pyx", "*.pxd", "*.jinja"} <= set(cython_exclusions)
    assert "examples/patch_antenna/output/*" in step_exclusions
    assert "examples/stl/Trinity_Alps.stl" in stl_exclusions
    assert "examples/stl/Stanford_Bunny.h5" in stl_exclusions
    assert "examples/stl/Stanford_Bunny.stl" not in stl_exclusions


@pytest.mark.unit
def test_source_distribution_uses_the_same_large_asset_exclusions():
    manifest = Path(__file__).resolve().parents[2] / "MANIFEST.in"
    directives = {
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "prune testing" in directives
    assert "prune reframe_tests" in directives
    assert "prune toolboxes/STEPtoVoxel/examples/patch_antenna/output" in directives
    assert "prune toolboxes/STLtoVoxel/examples/stl/point_cloud" in directives
    assert "global-exclude *.so *.pyd" in directives

    for path in EXCLUDED_PACKAGE_DATA["toolboxes.STLtoVoxel"]:
        if "*" not in path:
            assert f"exclude toolboxes/STLtoVoxel/{path}" in directives
