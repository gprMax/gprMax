"""Real multi-rank geometry-import and SAR integration regression."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax


def _mpi_launcher():
    """Prefer the launcher installed beside the active Python/mpi4py."""

    environment_launcher = Path(sys.executable).parent / "mpiexec"
    if environment_launcher.is_file():
        return str(environment_launcher)
    return shutil.which("mpiexec")


def _write_tagged_geometry(tmp_path):
    geometry = tmp_path / "mpi_geometry"
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.04, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Material(er=4, se=0.2, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.016, 0.006, 0.006),
            p2=(0.024, 0.014, 0.014),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0, 0, 0),
            p2=(0.04, 0.02, 0.02),
            filename=str(geometry),
        )
    )
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=geometry,
        geometry_only=True,
        hide_progress_bars=True,
    )
    return geometry.with_suffix(".h5"), tmp_path / "mpi_geometry_materials.json"


def _run_command(command, tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "FI_PROVIDER": "shm",
            "MPI4PY_RC_FINALIZE": "0",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.integration
@pytest.mark.skipif(_mpi_launcher() is None, reason="mpiexec is not installed")
def test_mpi_geometry_read_and_sar_match_serial_across_rank_boundary(tmp_path):
    """A tagged import and its Yee-edge DFT must survive an actual x split."""

    geometry_file, database_file = _write_tagged_geometry(tmp_path)
    assert geometry_file.is_file()
    assert database_file.is_file()

    model = tmp_path / "mpi_sar.in"
    model.write_text(
        "\n".join(
            (
                "#title: MPI geometry-read SAR regression",
                "#domain: 0.04 0.02 0.02",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 2e-9",
                "#pml_cells: 2",
                "#geometry_objects_read: 0 0 0 mpi_geometry.h5 mpi_geometry_materials n",
                "#waveform: ricker 1 1e9 pulse",
                "#hertzian_dipole: z 0.008 0.01 0.01 pulse",
                "#sar: 1e9 1e9 1 pulse 1 nyquist dose target",
                "",
            )
        ),
        encoding="utf-8",
    )

    common = (sys.executable, "-m", "gprMax", str(model), "--hide-progress-bars")
    serial_output = tmp_path / "serial"
    mpi_output = tmp_path / "mpi"
    _run_command((*common, "-o", str(serial_output)), tmp_path)
    _run_command(
        (
            _mpi_launcher(),
            "-n",
            "2",
            *common,
            "--mpi",
            "2",
            "1",
            "1",
            "-o",
            str(mpi_output),
        ),
        tmp_path,
    )

    with h5py.File(serial_output.with_suffix(".h5"), "r") as serial_file:
        serial = serial_file["sar/dose"]
        serial_cells = serial["cell_indices"][...]
        serial_sar = serial["sar"][...]
    with h5py.File(mpi_output.with_suffix(".h5"), "r") as mpi_file:
        mpi = mpi_file["sar/dose"]
        mpi_cells = mpi["cell_indices"][...]
        mpi_sar = mpi["sar"][...]

    # The imported target spans the decomposition plane at global x index 10.
    assert np.any(serial_cells[:, 0] < 10)
    assert np.any(serial_cells[:, 0] >= 10)
    np.testing.assert_array_equal(mpi_cells, serial_cells)
    np.testing.assert_allclose(mpi_sar, serial_sar, rtol=2e-5, atol=1e-12)
    assert np.all(np.isfinite(mpi_sar))
