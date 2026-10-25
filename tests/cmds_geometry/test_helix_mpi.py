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

"""Serial/MPI parity for ``Helix``.

Each rank builds in its local frame and writes only its owned cells into a
global buffer; stitching the ranks must reproduce the serial ``solid`` array
exactly.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
SENTINEL = np.iinfo(np.uint32).max


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and an MPI launcher",
)
def test_mpi_partition_reproduces_serial_helix(tmp_path):
    environment = os.environ.copy()
    # The parent pytest session uses a serial-import MPI fallback. Let the
    # MPI launcher select its transport for the actual multi-rank solve.
    environment.pop("FI_PROVIDER", None)
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
        PRTE_MCA_rmaps_default_mapping_policy=":oversubscribe",
    )

    serial_output = tmp_path / "serial.npz"
    serial_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(serial_output),
    ]
    result = subprocess.run(
        serial_command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

    mpi_output = tmp_path / "mpi.npz"
    mpi_command = [
        MPIEXEC,
        "-n",
        "2",
        sys.executable,
        str(Path(__file__).resolve()),
        str(mpi_output),
        "--partition",
        "2",
        "1",
        "1",
    ]
    result = subprocess.run(
        mpi_command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

    serial = np.load(serial_output)
    reference = serial["solid"]
    assert np.count_nonzero(reference) > 0

    stitched = np.full(tuple(serial["global_size"]), SENTINEL, dtype=np.uint32)
    starts = []
    for rank in range(2):
        rank_data = np.load(mpi_output.with_suffix(f".rank{rank}.npz"))
        start = rank_data["start"]
        local = rank_data["solid"]
        assert np.all(start >= 0)
        assert np.all(start + np.asarray(local.shape) <= serial["global_size"])
        region = tuple(slice(int(start[axis]), int(start[axis]) + local.shape[axis]) for axis in range(3))
        existing = stitched[region]
        overlap = existing != SENTINEL
        assert np.array_equal(existing[overlap], local[overlap])
        stitched[region] = local
        starts.append(tuple(start))

    assert len(set(starts)) == 2
    assert not np.any(stitched == SENTINEL)
    np.testing.assert_array_equal(stitched, reference)


def _worker(args):
    import gprMax
    import gprMax.model as model_mod

    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    model_mod.Model.build = patched_build

    dl = 1e-3
    scene = gprMax.Scene()
    for item in (
        gprMax.Discretisation(p1=(dl, dl, dl)),
        gprMax.Domain(p1=(0.024, 0.024, 0.024)),
        gprMax.TimeWindow(iterations=1),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"),
        gprMax.Helix(
            p1=(0.012, 0.012, 0.004),
            p2=(0.012, 0.012, 0.020),
            coil_radius=5e-3,
            pitch=3e-3,
            turns=2.0,
            wire_radius=1.5e-3,
            handedness="r",
            averaging="n",
            material_id="diel",
        ),
    ):
        scene.add(item)

    try:
        gprMax.run(
            scenes=[scene],
            outputfile=args.output.with_suffix(""),
            geometry_only=True,
            mpi=tuple(args.partition) if args.partition else None,
            hide_progress_bars=True,
            log_level=40,
        )
    finally:
        model_mod.Model.build = original_build

    grid = captured["grid"]
    if getattr(grid, "is_distributed", False):
        offset = np.asarray(grid.negative_halo_offset, dtype=int)
        lower = np.asarray(grid.lower_extent, dtype=int)
        global_size = np.asarray(grid.global_size, dtype=int)
        owned = tuple(slice(int(offset[axis]), int(grid.size[axis])) for axis in range(3))
        start = lower + offset
    else:
        global_size = np.asarray(grid.solid.shape, dtype=int)
        owned = (slice(None), slice(None), slice(None))
        start = np.zeros(3, dtype=int)

    destination = args.output
    if args.partition:
        from mpi4py import MPI

        destination = args.output.with_suffix(f".rank{MPI.COMM_WORLD.rank}.npz")
    np.savez(
        destination,
        solid=np.asarray(grid.solid)[owned],
        start=start,
        global_size=global_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--partition", type=int, nargs=3)
    _worker(parser.parse_args())
