"""Real MPI regression for a hard voltage port on an internal rank face."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _mpi_launcher():
    environment_launcher = Path(sys.executable).parent / "mpiexec"
    if environment_launcher.is_file():
        return str(environment_launcher)
    return shutil.which("mpiexec")


def _run(command, directory):
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
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.integration
@pytest.mark.skipif(_mpi_launcher() is None, reason="mpiexec is not installed")
def test_hard_voltage_port_on_internal_rank_boundary_matches_serial(tmp_path):
    """The positive rank must use its halo, not mistake its face for x=0."""

    model = tmp_path / "hard_port.in"
    model.write_text(
        "\n".join(
            (
                "#title: hard voltage port at MPI split",
                "#domain: 0.04 0.02 0.02",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 1e-9",
                "#pml_cells: 2",
                "#waveform: ricker 1 1e9 pulse",
                # The 2x1x1 decomposition splits x at 0.02 m. For a z-directed
                # hard source, x is transverse to the Ampere current loop.
                "#voltage_source: z 0.02 0.01 0.01 0 pulse 0 1e-9 feed nyquist 50",
                "",
            )
        ),
        encoding="utf-8",
    )

    common = (sys.executable, "-m", "gprMax", str(model), "--hide-progress-bars")
    serial_output = tmp_path / "serial"
    mpi_output = tmp_path / "mpi"
    _run((*common, "-o", str(serial_output)), tmp_path)
    _run(
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

    with h5py.File(serial_output.with_suffix(".h5")) as serial, h5py.File(
        mpi_output.with_suffix(".h5")
    ) as mpi:
        for dataset in ("S11", "Zin", "Yin", "Vtotal", "Iloop"):
            reference = serial[f"ports/feed/{dataset}"][...]
            np.testing.assert_allclose(
                mpi[f"ports/feed/{dataset}"][...],
                reference,
                rtol=2e-5,
                atol=2e-6 * max(float(np.nanmax(np.abs(reference), initial=0.0)), 1e-18),
                equal_nan=True,
            )
