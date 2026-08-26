"""Real MPI regression for direct layered time-domain NTFF collection."""

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
    environment.pop("MPI4PY_RC_FINALIZE", None)
    if Path(command[0]).name.startswith(("mpiexec", "mpirun")):
        # The suite-level ``FI_PROVIDER=shm`` fallback permits an in-process
        # mpi4py import on some Linux installations, but forcing that provider
        # on a real MPICH launch can deadlock during finalisation. Let the
        # launcher select its available transport for the distributed run.
        environment.pop("FI_PROVIDER", None)
    elif sys.platform.startswith("linux"):
        environment.setdefault("FI_PROVIDER", "shm")
    environment.update(
        {
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
def test_layered_time_ntff_matches_serial_across_rank_boundary(tmp_path):
    model = tmp_path / "layered_time_mpi.in"
    model.write_text(
        "\n".join(
            (
                "#domain: 0.08 0.06 0.06",
                "#dx_dy_dz: 0.004 0.004 0.004",
                "#time_window: 8e-10",
                "#pml_cells: 2",
                "#material: 4 0 1 0 dielectric",
                "#box: 0 0 0 0.08 0.06 0.032 dielectric",
                "#waveform: ricker 1 3e9 pulse",
                "#hertzian_dipole: x 0.04 0.032 0.04 pulse",
                "#ntff_surface: 0.024 0.016 0.020 0.056 0.044 0.044 surface",
                "#ntff_layered_background: halfspace z free_space 0.032 dielectric",
                "#ntff_layered_time: surface transient halfspace 1e-11 100000",
                "#ntff_layered_time_far_field: 20 30 transient upper Ex Etheta Ephi Hphi",
                "#ntff_layered_time_far_field: 160 30 transient lower Ex Etheta Ephi Hphi",
                "",
            )
        ),
        encoding="utf-8",
    )

    common = (
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "--hide-progress-bars",
        "-cpu_precision",
        "double",
    )
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

    with h5py.File(serial_output.with_suffix(".h5"), "r") as serial, h5py.File(
        mpi_output.with_suffix(".h5"), "r"
    ) as mpi:
        for output_id in ("upper", "lower"):
            reference = serial[f"ntff/surface/time_far_field/{output_id}"]
            distributed = mpi[f"ntff/surface/time_far_field/{output_id}"]
            assert distributed.attrs["collection_backend"].startswith("mpi_")
            for dataset in (
                "times",
                "theta",
                "phi",
                "directions",
                "observation_material_index",
                "observation_impedance",
                "observation_wave_speed",
                "impulse_counts",
                "discarded_path_amplitude_sums",
            ):
                np.testing.assert_allclose(distributed[dataset][...], reference[dataset][...], rtol=2e-13, atol=2e-13)
            for component in ("Ex", "Etheta", "Ephi", "Hphi"):
                np.testing.assert_allclose(
                    distributed[f"fields/{component}"][...],
                    reference[f"fields/{component}"][...],
                    rtol=2e-12,
                    atol=1e-15,
                )
