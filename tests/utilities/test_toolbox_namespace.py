"""Installed and source toolboxes must not claim a generic Python namespace."""

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from packaging_config import distribution_packages

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_every_installed_package_is_under_gprmax():
    packages = distribution_packages()
    assert "gprMax.toolboxes" in packages
    assert all(name == "gprMax" or name.startswith("gprMax.") for name in packages)
    assert not (ROOT / "toolboxes").exists()


def test_toolbox_imports_do_not_depend_on_an_unqualified_toolboxes_package():
    for path in (ROOT / "gprMax/toolboxes").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "toolboxes" for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert (node.module or "").split(".")[0] != "toolboxes", path


@pytest.mark.parametrize("preload_foreign", [False, True])
def test_unrelated_toolboxes_package_does_not_shadow_gprmax(tmp_path, preload_foreign):
    foreign = tmp_path / "toolboxes"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("SENTINEL = 'not gprMax'\n", encoding="utf-8")
    script = (
        "import sys\n"
        + ("import toolboxes\nforeign = toolboxes\n" if preload_foreign else "")
        + "import gprMax.toolboxes as own\n"
        "from gprMax.toolboxes.SFCW.processing import load_receiver\n"
        "from gprMax.toolboxes.Optimisation import LocalExecutor\n"
        "from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500\n"
        "assert own.__name__ == 'gprMax.toolboxes'\n"
        "assert load_receiver.__module__ == 'gprMax.toolboxes.SFCW.processing'\n"
        + (
            "assert sys.modules['toolboxes'] is foreign\nassert foreign.SENTINEL == 'not gprMax'\n"
            if preload_foreign
            else "assert 'toolboxes' not in sys.modules\n"
        )
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "Plotting.plot_Ascan",
        "Plotting.plot_Bscan",
        "Plotting.plot_port",
        "Plotting.plot_source_wave",
        "Utilities.outputfiles_merge",
        "MaterialDatabase",
        "SFCW",
        "FMCW",
        "ImpulseResponse",
    ],
)
def test_namespaced_module_entry_points_ignore_foreign_toolboxes(tmp_path, module):
    foreign = tmp_path / "toolboxes"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("raise RuntimeError('wrong toolbox namespace')\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLCONFIGDIR=str(tmp_path / "mpl"), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", "gprMax.toolboxes." + module, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert "cd gprMax;" not in result.stdout
