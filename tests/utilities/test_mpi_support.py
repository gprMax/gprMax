"""Tests for the deferred MPI runtime boundary."""

import os
import subprocess
import sys
import textwrap

import pytest

from gprMax.mpi_support import MPIUnavailableError, require_mpi


@pytest.mark.unit
def test_serial_run_does_not_import_mpi4py(tmp_path):
    """A serial geometry build must not initialise an MPI runtime."""

    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        class BlockMPI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mpi4py" or fullname.startswith("mpi4py."):
                    raise ModuleNotFoundError("mpi4py import was not deferred")
                return None

        sys.meta_path.insert(0, BlockMPI())
        import gprMax

        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
        scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-12))
        scene.add(
            gprMax.Box(
                p1=(0.002, 0.002, 0.002),
                p2=(0.008, 0.008, 0.008),
                material_id="pec",
            )
        )
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=os.environ["GPRMAX_TEST_OUTPUT"],
            hide_progress_bars=True,
        )

        assert not any(
            name == "mpi4py" or name.startswith("mpi4py.")
            for name in sys.modules
        )
        """
    )
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["GPRMAX_TEST_OUTPUT"] = str(tmp_path / "serial_without_mpi")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_require_mpi_returns_real_mpi_module():
    MPI = pytest.importorskip("mpi4py.MPI")

    assert require_mpi("test MPI feature") is MPI


@pytest.mark.unit
def test_require_mpi_reports_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)

    with pytest.raises(MPIUnavailableError, match="working MPI runtime.*mpi4py") as excinfo:
        require_mpi("distributed test feature")

    assert isinstance(excinfo.value.__cause__, ImportError)
