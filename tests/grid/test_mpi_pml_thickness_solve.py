"""Real MPI coverage of rank-local PML validation through the public API."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).with_name("mpiexec.exe" if os.name == "nt" else "mpiexec")
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
ERROR = "PML has too many cells for the domain size"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
        reason="requires mpi4py and an MPI launcher",
    ),
]


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("thickness", (4, 13), ids=("valid", "locally-invalid"))
def test_three_rank_boundary_pml_validation(tmp_path, axis, thickness):
    output = tmp_path / "model"
    environment = os.environ.copy()
    # Do not inherit the parent unit-test session's serial MPI import settings.
    environment.pop("FI_PROVIDER", None)
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    command = [
        MPIEXEC,
        "-n",
        "3",
        sys.executable,
        str(Path(__file__).resolve()),
        str(output),
        "--axis",
        str(axis),
        "--thickness",
        str(thickness),
    ]
    try:
        result = subprocess.run(
            command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"MPI PML validation did not finish within 60 s:\n{exc.stdout}\n{exc.stderr}")

    log = result.stdout + result.stderr
    if thickness == 4:
        assert result.returncode == 0, log
        assert "Traceback" not in log, log
        assert output.with_suffix(".h5").is_file(), log
        expected_status = "built"
    else:
        assert result.returncode != 0, log
        assert ERROR in log, log
        assert not output.with_suffix(".h5").exists(), log
        expected_status = "rejected"

    coordinates = set()
    for rank in range(3):
        record_path = output.with_suffix(f".rank{rank}.json")
        # Global input validation or an import failure cannot satisfy this:
        # each rank must have reached the real MPIGrid.build().
        assert record_path.is_file(), log
        record = json.loads(record_path.read_text())
        coordinate = record["coords"][axis]
        coordinates.add(coordinate)
        assert record["global_size"] == [36, 36, 36]
        assert record["size"][axis] == (12 if coordinate == 0 else 13)
        expected_pmls = dict.fromkeys(("x0", "xmax", "y0", "ymax", "z0", "zmax"), 0)
        if coordinate == 0:
            expected_pmls[f"{'xyz'[axis]}0"] = thickness
        assert record["pmls"] == expected_pmls
        assert record["status"] == expected_status, record
        if expected_status == "rejected":
            assert record["error"] == ERROR, record
    assert coordinates == {0, 1, 2}


def _worker(args):
    import gprMax
    from gprMax.grid.mpi_grid import MPIGrid

    original_build = MPIGrid.build

    def record_build(grid):
        path = args.output.with_suffix(f".rank{grid.rank}.json")
        record = {
            "coords": list(grid.coords),
            "global_size": grid.global_size.tolist(),
            "size": grid.size.tolist(),
            "pmls": dict(grid.pmls["thickness"]),
            "status": "entered",
        }
        path.write_text(json.dumps(record))
        try:
            original_build(grid)
        except ValueError as exc:
            record.update(status="rejected", error=str(exc))
            path.write_text(json.dumps(record))
            # Every rank must receive the coordinated validation error. Wait
            # for their records before MPIContext catches it and calls Abort.
            # A rank diverging into other build work is caught by the timeout.
            grid.comm.Barrier()
            raise
        record["status"] = "built"
        path.write_text(json.dumps(record))

    MPIGrid.build = record_build
    partition = [1, 1, 1]
    partition[args.axis] = 3
    thickness = [0] * 6
    thickness[args.axis] = args.thickness
    scene = gprMax.Scene()
    for item in (
        gprMax.Discretisation(p1=(0.002, 0.002, 0.002)),
        gprMax.Domain(p1=(0.072, 0.072, 0.072)),
        # Both cases are globally valid: 4 < 36 and 13 < 36. Only the
        # owning rank's 12-cell partition is too small for the latter.
        gprMax.PMLThickness(thickness=tuple(thickness)),
        gprMax.TimeWindow(iterations=8),
        gprMax.OMPThreads(n=1),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"),
        gprMax.HertzianDipole(p1=(0.036, 0.036, 0.036), polarisation="z", waveform_id="pulse"),
        gprMax.Rx(p1=(0.040, 0.040, 0.040)),
    ):
        scene.add(item)
    gprMax.run(
        scenes=[scene],
        mpi=tuple(partition),
        outputfile=args.output,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", type=int, choices=range(3), required=True)
    parser.add_argument("--thickness", type=int, choices=(4, 13), required=True)
    _worker(parser.parse_args())
