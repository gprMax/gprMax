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

"""MPI domain decomposition of placed and rotated antenna models matches serial runs.

The antenna sits at the centre of the domain, so a 2 x 2 x 1 decomposition
puts rank boundaries through the middle of every antenna, including its feeds.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
PARTITION = (2, 2, 1)
# model name: (resolution, domain, placement, iterations)
CASES = {
    "MALA_1200": (0.002, (0.240, 0.240, 0.100), (0.120, 0.120, 0.030), 300),
    "GSSI_1500": (0.002, (0.240, 0.240, 0.100), (0.120, 0.120, 0.030), 300),
    "GSSI_400": (0.002, (0.360, 0.360, 0.240), (0.180, 0.180, 0.030), 300),
}
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
        reason="requires mpi4py and an MPI launcher",
    ),
]


def _run(output, model, rotation, partition=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), str(output), model, str(rotation)]
    if partition is not None:
        command = [
            MPIEXEC,
            "-n",
            str(np.prod(partition)),
            *command,
            "--partition",
            *map(str, partition),
        ]
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    result = subprocess.run(
        command, cwd=output.parent, env=environment, capture_output=True, text=True, timeout=900
    )
    output.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    with h5py.File(output.with_suffix(".h5")) as data:
        group = data["rxs/rx1"]
        (component,) = [key for key in group if key.startswith("E")]
        return component, group[component][...], tuple(group.attrs["GridPosition"])


@pytest.mark.slow
@pytest.mark.parametrize("rotation", (0, 90))
@pytest.mark.parametrize("model", tuple(CASES))
def test_mpi_decomposition_matches_serial(tmp_path, model, rotation):
    serial = _run(tmp_path / "serial", model, rotation)
    decomposed = _run(tmp_path / "mpi", model, rotation, PARTITION)
    assert decomposed[0] == serial[0] == ("Ey" if rotation == 0 else "Ex")
    assert decomposed[2] == serial[2]
    scale = np.max(np.abs(serial[1]))
    assert np.all(np.isfinite(serial[1])) and scale > 0
    np.testing.assert_allclose(decomposed[1], serial[1], rtol=0, atol=2e-12 * scale)


def _worker(args):
    import gprMax
    from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400, antenna_like_GSSI_1500
    from gprMax.toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200

    factory = {
        "MALA_1200": antenna_like_MALA_1200,
        "GSSI_1500": antenna_like_GSSI_1500,
        "GSSI_400": antenna_like_GSSI_400,
    }[args.model]
    dl, domain, placement, iterations = CASES[args.model]
    scene = gprMax.Scene()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        objects = factory(*placement, resolution=dl, rotation=args.rotation)
    for obj in (
        gprMax.Domain(p1=domain),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.TimeWindow(iterations=iterations),
        *objects,
    ):
        scene.add(obj)
    gprMax.run(
        scenes=[scene],
        outputfile=Path(args.output),
        mpi=tuple(args.partition) if args.partition is not None else None,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("model", choices=tuple(CASES))
    parser.add_argument("rotation", type=int)
    parser.add_argument("--partition", nargs=3, type=int)
    args = parser.parse_args()
    try:
        _worker(args)
    except BaseException:
        if args.partition is not None:
            from mpi4py import MPI

            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
