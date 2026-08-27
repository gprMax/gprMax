# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

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


@pytest.mark.unit
def test_requesting_mpi_features_without_dependency_reports_actionable_error(tmp_path):
    """Public MPI paths must fail at the optional-dependency boundary."""

    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockMPI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mpi4py" or fullname.startswith("mpi4py."):
                    raise ModuleNotFoundError("blocked optional mpi4py dependency")
                return None

        sys.meta_path.insert(0, BlockMPI())
        import gprMax
        from gprMax.mpi_support import MPIUnavailableError

        cases = (
            ({"mpi": (1, 1, 1)}, "MPI domain decomposition"),
            ({"taskfarm": True}, "MPI task farming"),
        )
        for options, feature in cases:
            scene = gprMax.Scene()
            try:
                gprMax.run(
                    scenes=[scene],
                    n=1,
                    outputfile="unused",
                    hide_progress_bars=True,
                    **options,
                )
            except MPIUnavailableError as exc:
                assert f"{feature} requires a working MPI runtime" in str(exc)
            else:
                raise AssertionError(f"{feature} unexpectedly passed without mpi4py")
        """
    )
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
