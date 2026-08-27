"""Tests for installed, version-matched example resources."""

import pytest

from gprMax import examples
from gprMax._version import __version__


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
