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

"""Real serial/MPI parity for all SAR and radiometry temporal windows."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import gprMax
from .test_radiometry import _lossy_scene


ROOT = Path(__file__).resolve().parents[2]
MPIEXEC = shutil.which("mpiexec", path=str(Path(sys.executable).parent)) or shutil.which("mpiexec")


def _run_model(path, distributed=False):
    scene = _lossy_scene()
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="lossy"))
    for window in ("rectangular", "hann", "hann_end"):
        for kind in (gprMax.SAR, gprMax.Radiometry):
            scene.add(
                kind(
                    frequencies=(1e9, 1.25e9),
                    tags="target",
                    waveform_id="pulse",
                    window=window,
                    id=f"{kind.__name__.lower()}_{window}",
                )
            )
    options = {"mpi": (1, 2, 1)} if distributed else {}
    gprMax.run(scenes=[scene], outputfile=path, cpu_precision="double", hide_progress_bars=True, **options)


@pytest.mark.integration
@pytest.mark.skipif(MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="requires mpi4py and mpiexec")
def test_sar_radiometry_windows_match_across_rank_boundary(tmp_path):
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    common = (sys.executable, "-m", "tests.outputs.test_sar_windows_mpi")
    for distributed, name in ((False, "serial"), (True, "mpi")):
        env = environment.copy()
        command = (*common, str(tmp_path / name))
        if distributed:
            # Do not force the suite's in-process shm import workaround on a
            # complete MPI application; let its launcher select the transport.
            if env.get("FI_PROVIDER") == "shm":
                env.pop("FI_PROVIDER")
            command = (MPIEXEC, "-n", "2", *command, "--mpi")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / "serial.h5") as serial, h5py.File(tmp_path / "mpi.h5") as mpi:
        for kind in ("sar", "radiometry"):
            for window in ("rectangular", "hann", "hann_end"):
                path = f"{kind}/{kind}_{window}"
                expected, actual = serial[path], mpi[path]
                cells = expected["cell_indices"][...]
                assert np.any(cells[:, 1] < 6) and np.any(cells[:, 1] >= 6)
                np.testing.assert_array_equal(actual["cell_indices"], cells)
                assert actual.attrs["Window"] == window
                if window == "hann_end":
                    assert actual.attrs["WindowTaperFraction"] == 0.2
                quantity = "sar" if kind == "sar" else "normalised_absorption_density"
                for field in (quantity, "absorbed_power_density", "source_spectrum", "normalisation_scale"):
                    values = expected[field][...]
                    assert np.all(np.isfinite(values))
                    np.testing.assert_allclose(
                        actual[field][...], values, rtol=2e-12, atol=2e-12 * np.max(np.abs(values))
                    )


if __name__ == "__main__":
    _run_model(sys.argv[1], distributed="--mpi" in sys.argv[2:])
