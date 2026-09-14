"""Keep hosted documentation and its isolated CI check in agreement."""

from pathlib import Path
import runpy
import sys

import pytest
import yaml


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path):
    # Preserve GitHub Actions' "on" key instead of treating it as a boolean.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_sphinx_configuration_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    config = runpy.run_path(str(ROOT / "docs/source/conf.py"))
    assert config["ROOT"] == ROOT
    assert sys.path[0] == str(ROOT)
    version = runpy.run_path(str(ROOT / "gprMax/_version.py"))["__version__"]
    assert config["version"] == version


def test_documentation_ci_matches_readthedocs_environment():
    rtd = read_yaml(ROOT / ".readthedocs.yaml")
    workflow = read_yaml(ROOT / ".github/workflows/docs.yml")
    job = workflow["jobs"]["docs"]
    assert job["runs-on"] == rtd["build"]["os"]
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == rtd["build"]["tools"]["python"]
    assert rtd["python"]["install"] == [{"requirements": "docs/requirements.txt"}]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "python -m pip install -r docs/requirements.txt" in commands
    assert "python -m pip check" in commands
    assert "-r requirements.txt" not in commands
    assert "pip install -e" not in commands
    assert "pip install ." not in commands
    for package in rtd["build"]["apt_packages"]:
        assert f"sudo apt-get install --yes {package}" in commands


def test_documentation_builds_reject_warnings_and_cover_both_formats():
    rtd = read_yaml(ROOT / ".readthedocs.yaml")
    workflow = read_yaml(ROOT / ".github/workflows/docs.yml")
    assert rtd["sphinx"]["fail_on_warning"] == "true"
    assert rtd["sphinx"]["configuration"] == "docs/source/conf.py"
    assert "pdf" in rtd["formats"]
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["docs"]["steps"])
    for builder in ("html", "latex"):
        assert f"-b {builder} -aE -W --keep-going docs/source" in commands


def test_documentation_ci_runs_for_code_changes_not_only_documentation():
    workflow = read_yaml(ROOT / ".github/workflows/docs.yml")
    # The regression came from a toolbox import change, not a docs edit.
    assert "pull_request" in workflow["on"]
    assert not workflow["on"]["pull_request"]
    assert "master" in workflow["on"]["push"]["branches"]
    assert "paths" not in workflow["on"]["push"]
    assert "paths-ignore" not in workflow["on"]["push"]
