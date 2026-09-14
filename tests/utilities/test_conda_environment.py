"""The default Conda name identifies the release series, not a Git branch."""

import ast
from pathlib import Path
import re

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_default_conda_environment_tracks_major_version():
    version_module = ast.parse((ROOT / "gprMax" / "_version.py").read_text())
    version = next(
        ast.literal_eval(node.value)
        for node in version_module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    )
    environment = (ROOT / "conda_env.yml").read_text()
    name = re.search(r"^name:\s*(\S+)\s*$", environment, re.MULTILINE)
    assert name is not None
    assert name.group(1) == f"gprMax-v{version.split('.')[0]}"
