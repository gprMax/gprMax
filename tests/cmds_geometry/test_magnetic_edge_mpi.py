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

"""Serial/MPI parity for a magnetic edge lying on an in-plane partition boundary."""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = pytest.mark.integration
INF = float("inf")
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).with_name("mpiexec.exe" if os.name == "nt" else "mpiexec")
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")


def _scene(mode, wall_cell):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.032, 0.016, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(iterations=400))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="w"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z" if mode == "TM" else "y",
            p1=(0.008, 0.008, INF),
            waveform_id="w",
        )
    )
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z" if mode == "TM" else "y",
            p1=(0.024, 0.008, INF),
            waveform_id="w",
        )
    )
    field = "Ez" if mode == "TM" else "Ey"
    scene.add(gprMax.Rx(p1=(0.012, 0.008, INF), outputs=[field], id="near"))
    scene.add(gprMax.Rx(p1=(0.020, 0.008, INF), outputs=[field], id="far"))
    x = wall_cell * 1e-3
    if mode == "TM":
        scene.add(gprMax.MagneticEdge(p1=(x, 0, INF), p2=(x, 0.016, INF), material_id="pmc"))
    else:
        for j in range(16):
            scene.add(
                gprMax.MagneticEdge(
                    p1=(x, j * 1e-3, INF), p2=(x, j * 1e-3, INF), material_id="pmc"
                )
            )
    return scene


def _run(output, mode, wall_cell, partition=None):
    options = {"mpi": tuple(partition)} if partition else {}
    gprMax.run(
        scenes=[_scene(mode, wall_cell)],
        outputfile=output,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
        **options,
    )


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and mpiexec",
)
@pytest.mark.parametrize("wall_cell", (15, 16))
@pytest.mark.parametrize("mode", ("TM", "TE"))
def test_magnetic_edge_on_partition_boundary_matches_serial(tmp_path, mode, wall_cell):
    reference, distributed = tmp_path / "serial", tmp_path / "mpi"
    _run(reference, mode, wall_cell)

    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPI4PY_RC_INITIALIZE="1",
    )
    result = subprocess.run(
        [
            MPIEXEC,
            "-n",
            "2",
            sys.executable,
            str(Path(__file__).resolve()),
            str(distributed),
            "--mode",
            mode,
            "--wall-cell",
            str(wall_cell),
            "--partition",
            "2",
            "1",
            "1",
        ],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with h5py.File(reference.with_suffix(".h5")) as expected, h5py.File(
        distributed.with_suffix(".h5")
    ) as actual:
        for rx in expected["rxs"]:
            for component in expected[f"rxs/{rx}"]:
                values = expected[f"rxs/{rx}/{component}"][:]
                probe = actual[f"rxs/{rx}/{component}"][:]
                scale = np.max(np.abs(values))
                assert scale > 1e-8
                np.testing.assert_allclose(probe, values, rtol=0, atol=scale * 3e-11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("TM", "TE"), required=True)
    parser.add_argument("--wall-cell", type=int, required=True)
    parser.add_argument("--partition", type=int, nargs=3, required=True)
    args = parser.parse_args()
    _run(args.output, args.mode, args.wall_cell, args.partition)
