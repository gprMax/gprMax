"""Real MPI/serial parity when normal variation distinguishes sampling planes."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and an MPI launcher",
)
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("split", ("normal", "transverse"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
def test_entrance_calibration_and_fields_match_serial(tmp_path, axis, split, formulation):
    partition = [1, 1, 1]
    partition[axis if split == "normal" else (axis + 1) % 3] = 2
    environment = os.environ.copy()
    # The parent pytest session uses a serial-import OFI fallback. Let the
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
    outputs = (tmp_path / "serial", tmp_path / "mpi")
    for output, distributed in zip(outputs, (False, True)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(output),
            "--axis",
            str(axis),
            "--formulation",
            formulation,
        ]
        if distributed:
            command = [MPIEXEC, "-n", "2", *command, "--partition", *map(str, partition)]
        result = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr

    serial = json.loads(outputs[0].with_suffix(".rank0.json").read_text())
    seen = set()
    for rank in range(2):
        distributed = json.loads(outputs[1].with_suffix(f".rank{rank}.json").read_text())
        for face, values in distributed.items():
            assert values == pytest.approx(serial[face]), face
            seen.add(face)
    assert seen == set(serial)
    # The deliberately different material at the outer end must not be sampled.
    assert serial["xyz"[axis] + "0"][:2] == [15.0, 2.0]
    assert serial["xyz"[axis] + "max"][:2] == [15.0, 2.0]
    with h5py.File(outputs[0].with_suffix(".h5")) as reference, h5py.File(
        outputs[1].with_suffix(".h5")
    ) as actual:
        for component in COMPONENTS:
            expected = reference[f"rxs/rx1/{component}"][...]
            result = actual[f"rxs/rx1/{component}"][...]
            assert np.isfinite(expected).all() and np.isfinite(result).all()
            scale = np.max(np.abs(expected))
            np.testing.assert_allclose(result, expected, rtol=0, atol=max(1e-14, scale * 2e-12))


def _worker(args):
    import gprMax
    from gprMax.pml import PML

    calibration = {}
    original = PML.calculate_update_coeffs

    def capture(self, er, mr):
        original(self, er, mr)
        calibration[self.ID] = [er, mr, self.CFS[0].sigma.max]

    PML.calculate_update_coeffs = capture
    dl = 0.002
    size = np.array((24, 26, 28))
    domain = tuple(size * dl)
    scene = gprMax.Scene()
    for item in (
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.Domain(p1=domain),
        gprMax.PMLThickness(thickness=4),
        gprMax.PMLFormulation(formulation=args.formulation),
        gprMax.TimeWindow(iterations=240),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=15, se=0, mr=2, sm=0, id="entrance"),
        gprMax.Material(er=4, se=0, mr=1, sm=0, id="outer"),
        gprMax.Box(p1=(0, 0, 0), p2=domain, material_id="entrance"),
        gprMax.Waveform(wave_type="gaussiandotnorm", amp=1, freq=1e9, id="pulse"),
        gprMax.HertzianDipole(p1=(0.024, 0.024, 0.024), polarisation="z", waveform_id="pulse"),
        # Excite every receiver component, including Hz, above roundoff.
        gprMax.HertzianDipole(p1=(0.020, 0.030, 0.030), polarisation="x", waveform_id="pulse"),
        gprMax.Rx(p1=(0.020, 0.022, 0.026)),
    ):
        scene.add(item)
    # Normal variation is artificial: it exposes a sampling mismatch, not
    # a proposed way to improve absorption. Each entrance voxel stays in host.
    lower_end = np.array(domain)
    lower_end[args.axis] = 3 * dl
    upper_start = np.zeros(3)
    upper_start[args.axis] = domain[args.axis] - 3 * dl
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(lower_end), material_id="outer"))
    scene.add(gprMax.Box(p1=tuple(upper_start), p2=domain, material_id="outer"))
    try:
        gprMax.run(
            scenes=[scene],
            outputfile=args.output,
            cpu_precision="double",
            mpi=tuple(args.partition) if args.partition else None,
            hide_progress_bars=True,
            log_level=40,
        )
    finally:
        PML.calculate_update_coeffs = original
    rank = 0
    if args.partition:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.rank
    args.output.with_suffix(f".rank{rank}.json").write_text(json.dumps(calibration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", type=int, required=True)
    parser.add_argument("--formulation", choices=("HORIPML", "MRIPML"), required=True)
    parser.add_argument("--partition", type=int, nargs=3)
    _worker(parser.parse_args())
