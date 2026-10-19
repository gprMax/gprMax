"""Grass voxels must be identical to a serial run under real MPI decomposition."""

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

CASES = {
    "yplus": dict(
        box=((0.06, 0.06, 0.02), (0.08, 0.08, 0.045)),
        grass=((0.06, 0.08, 0.02), (0.08, 0.08, 0.045)),
        limits=(0.075, 0.09),
    ),
    "xplus": dict(
        box=((0.06, 0.06, 0.02), (0.08, 0.08, 0.045)),
        grass=((0.08, 0.06, 0.02), (0.08, 0.08, 0.045)),
        limits=(0.075, 0.09),
    ),
    "zplus": dict(
        box=((0.06, 0.06, 0.02), (0.08, 0.08, 0.045)),
        grass=((0.06, 0.06, 0.045), (0.08, 0.08, 0.045)),
        limits=(0.075, 0.09),
    ),
    # The grass column starts below the box top, so rank boundaries can fall
    # between a blade's root and its tip.
    "ycross": dict(
        box=((0.06, 0.04, 0.02), (0.08, 0.06, 0.045)),
        grass=((0.06, 0.06, 0.02), (0.08, 0.06, 0.045)),
        limits=(0.06, 0.075),
    ),
}


def _scene(output, case, seed=17):
    import gprMax

    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.Domain(p1=(0.1,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.TimeWindow(iterations=10),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=8, se=0, mr=1, sm=0, id="half_space"),
    ):
        scene.add(obj)
    scene.add(
        gprMax.FractalBox(
            p1=case["box"][0],
            p2=case["box"][1],
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=1,
            mixing_model_id="half_space",
            id="my_fractal_box",
            seed=42,
        )
    )
    grass = dict(
        p1=case["grass"][0],
        p2=case["grass"][1],
        frac_dim=1.5,
        limits=case["limits"],
        n_blades=100,
        fractal_box_id="my_fractal_box",
    )
    if seed is not None:
        grass["seed"] = seed
    scene.add(gprMax.AddGrass(**grass))
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            filename="geometry_dump",
        )
    )
    return scene


def _dump(self, outputdir, name):
    self.grid_view.initialise_materials(include_solid=True)
    ID = self.grid_view.map_to_view_materials(self.grid_view.get_ID())
    data = self.grid_view.map_to_view_materials(self.grid_view.get_solid()).astype(np.int16)
    info = {"seed": ""}
    if hasattr(self.grid_view, "comm"):
        slices = self.grid_view.get_3d_output_slice()
        info.update(
            start=np.array([section.start for section in slices], dtype=np.int64),
            global_size=np.array(self.grid_view.global_size, dtype=np.int64),
        )
    surface = next(
        (
            surface
            for volume in self.grid.fractalvolumes
            for surface in volume.fractalsurfaces
            if surface.ID == "grass"
        ),
        None,
    )
    if surface is not None:
        # Blade tops are absolute coordinates, so the maps from every rank
        # must agree wherever the decomposition cuts through the blades.
        info["seed"] = "" if surface.seed is None else str(surface.seed)
        info["heights"] = surface.fractalsurface
    np.savez(outputdir / f"{name}.npz", data=data, ID=ID, **info)


def _environment():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        MPI4PY_RC_INITIALIZE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    return env


def _launch(command):
    result = subprocess.run(
        command, env=_environment(), cwd=ROOT, text=True, capture_output=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run(outputdir, case, partition=None, seed=17):
    import gprMax
    from gprMax.geometry_outputs.geometry_objects import GeometryObject, MPIGeometryObject

    outputdir.mkdir(parents=True, exist_ok=True)
    # The solid map is dumped to npz because the test only needs cell values;
    # this also keeps the test independent of parallel HDF5.
    GeometryObject.write_hdf5 = lambda self, title, pbar: _dump(self, outputdir, "serial")
    MPIGeometryObject.write_hdf5 = lambda self, title, pbar: _dump(
        self, outputdir, f"rank_{self.grid_view.comm.rank}"
    )
    gprMax.run(
        scenes=[_scene(outputdir, case, seed)],
        geometry_only=True,
        mpi=partition,
        outputfile=str(outputdir / "run"),
        hide_progress_bars=True,
        log_level=30,
    )


def _reconstruct(serial_dir, distributed_dir):
    serial = np.load(serial_dir / "serial.npz")["data"]
    recon = np.full(serial.shape, -1, dtype=np.int16)
    for path in sorted(distributed_dir.glob("rank_*.npz")):
        rank = np.load(path)
        start = rank["start"]
        section = tuple(
            slice(int(start[axis]), int(start[axis]) + rank["data"].shape[axis]) for axis in range(3)
        )
        recon[section] = rank["data"]
    return serial, recon


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires MPI and mpi4py",
)
@pytest.mark.parametrize(
    "case, partition",
    [
        ("yplus", (2, 2, 2)),
        ("xplus", (2, 2, 2)),
        ("zplus", (1, 1, 4)),
        ("ycross", (1, 3, 1)),
    ],
)
def test_grass_geometry_matches_serial_under_mpi(tmp_path, case, partition):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    script = str(Path(__file__).resolve())
    for output, mpi in ((serial, None), (distributed, partition)):
        command = [sys.executable, script, str(output), "--case", case]
        if mpi is not None:
            command = [MPIEXEC, "-n", str(int(np.prod(mpi))), *command, "--partition"]
            command += [str(value) for value in mpi]
        _launch(command)
    expected, actual = _reconstruct(serial, distributed)
    assert actual.min() >= 0, "some cells were not covered by the decomposition"
    assert np.array_equal(expected, actual)


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires MPI and mpi4py",
)
def test_grass_geometry_is_shared_across_ranks_without_seed(tmp_path):
    distributed = tmp_path / "mpi"
    script = str(Path(__file__).resolve())
    partition = (1, 3, 1)
    command = [MPIEXEC, "-n", str(int(np.prod(partition))), sys.executable, script]
    command += [str(distributed), "--case", "ycross", "--no-seed"]
    command += ["--partition", *[str(value) for value in partition]]
    _launch(command)

    dumps = [np.load(path) for path in sorted(distributed.glob("rank_*.npz"))]
    seeds = {str(dump["seed"].item()) for dump in dumps}
    assert len(seeds) == 1, f"ranks resolved different grass seeds: {sorted(seeds)}"
    seed = seeds.pop()
    assert seed, "no seed was shared between the ranks"
    # Every rank builds the whole grass surface, so the blade height maps
    # must agree even where the decomposition cuts through the blades.
    assert all(np.array_equal(dump["heights"], dumps[0]["heights"]) for dump in dumps)

    serial = tmp_path / "serial"
    command = [sys.executable, script, str(serial), "--case", "ycross", "--seed", seed]
    _launch(command)
    expected, actual = _reconstruct(serial, distributed)
    assert actual.min() >= 0, "some cells were not covered by the decomposition"
    assert np.array_equal(expected, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outputdir", type=Path)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--partition", type=int, nargs=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()
    _run(
        args.outputdir,
        CASES[args.case],
        tuple(args.partition) if args.partition else None,
        None if args.no_seed else args.seed,
    )
