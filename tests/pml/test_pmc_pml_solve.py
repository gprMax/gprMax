"""Physical mirror, late-time and cross-backend PMC/PML regressions.

An upper PMC must give the same response as the reflected lower PMC, not
merely avoid NaNs. Receivers sample both E tangent and H normal to the
mirror, whose Yee coordinates and signs are unchanged by that reflection.
"""

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
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).with_name("mpiexec.exe" if os.name == "nt" else "mpiexec")
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
PAIRS = [(stretch, wall) for stretch in range(3) for wall in range(3) if wall != stretch]


def _scene(
    stretch=2,
    wall=1,
    upper=True,
    formulation="HORIPML",
    order=1,
    material="free_space",
    both=False,
    steps=800,
    internal=False,
):
    counts = np.array((8, 8, 8))
    counts[stretch] = 48
    spacing = np.array((0.04, 0.05, 0.06))
    domain = counts * spacing
    polarisation = next(a for a in range(3) if a not in (stretch, wall))
    thickness = [0] * 6
    if not internal:
        thickness[stretch] = thickness[stretch + 3] = 6
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(domain)),
        gprMax.Discretisation(p1=tuple(spacing)),
        gprMax.OMPThreads(n=1),
        gprMax.PMLThickness(thickness=tuple(thickness)),
        gprMax.PMLFormulation(formulation=formulation),
        gprMax.TimeWindow(iterations=steps),
        gprMax.Waveform(wave_type="gauspulse", amp=0.001, freq=150e6, id="w"),
    ):
        scene.add(obj)
    if internal:
        # Terminate inside the grid, exercising the extra electric terminal
        # plane as well as the transverse PMC plane. Other slab faces reach
        # the domain walls, so they retain the declared PEC/PMC boundaries.
        for side, (start, stop) in enumerate(((3, 9), (39, 45))):
            p1, p2 = np.zeros(3), domain.copy()
            p1[stretch], p2[stretch] = start * spacing[stretch], stop * spacing[stretch]
            scene.add(
                gprMax.PMLSlab(
                    p1=tuple(p1),
                    p2=tuple(p2),
                    maximum_face="xyz"[stretch] + ("max" if side else "0"),
                )
            )
    if material != "free_space":
        scene.add(gprMax.Material(er=2, se=0.002, mr=1, sm=0, id="host"))
        if material == "debye":
            scene.add(
                gprMax.AddDebyeDispersion(
                    poles=1, er_delta=[1.0], tau=[1e-9], material_ids=["host"]
                )
            )
        elif material == "lorentz":
            scene.add(
                gprMax.AddLorentzDispersion(
                    poles=1, er_delta=[1.0], omega=[600e6], delta=[50e6], material_ids=["host"]
                )
            )
        scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(domain), material_id="host"))
    for axis in range(3):
        if axis == stretch:
            continue
        for side in (0, 1):
            pmc = axis == wall and (both or side == int(upper))
            scene.add(
                gprMax.SymmetryBoundary(
                    face="xyz"[axis] + ("max" if side else "0"), type="pmc" if pmc else "pec"
                )
            )
    if order == 2:
        for pole in range(2):
            scene.add(
                gprMax.PMLCFS(
                    alphascalingprofile="constant",
                    alphascalingdirection="forward",
                    alphamin=0.2 if pole else 0.001,
                    alphamax=0.2 if pole else 0.001,
                    kappascalingprofile="linear",
                    kappascalingdirection="forward",
                    kappamin=1,
                    kappamax=1.2 + 0.3 * pole,
                    sigmascalingprofile="linear",
                    sigmascalingdirection="forward",
                    sigmamin=0,
                    sigmamax=0.1 * (pole + 1),
                )
            )
    source = np.array((3, 3, 3))
    source[stretch] = 28
    source[wall] = 2 if not upper else counts[wall] - 2
    scene.add(
        gprMax.HertzianDipole(
            polarisation="xyz"[polarisation], p1=tuple(source * spacing), waveform_id="w"
        )
    )
    components = ["E" + "xyz"[polarisation], "H" + "xyz"[wall]]
    # Include the PMC/PML intersection itself, not just a remote receiver.
    probes = (
        ((22, False), (8, True), (40, True)) if internal else ((22, False), (2, True), (46, True))
    )
    for index, (distance, on_wall) in enumerate(probes):
        rx = source.copy()
        rx[stretch] = distance
        if on_wall:
            rx[wall] = counts[wall] if upper else 0
        scene.add(gprMax.Rx(p1=tuple(rx * spacing), outputs=components, id=f"rx{index}"))
    return scene


def _run(path, precision="double", options=None, **scene_options):
    gprMax.run(
        scenes=[_scene(**scene_options)],
        outputfile=path,
        cpu_precision=precision,
        gpu_precision=precision,
        hide_progress_bars=True,
        log_level=40,
        **(options or {}),
    )


def _compare(first, second, precision="double"):
    maximum = 0
    with h5py.File(first.with_suffix(".h5")) as a, h5py.File(second.with_suffix(".h5")) as b:
        assert a.attrs["dt"] == b.attrs["dt"]
        for rx in a["rxs"]:
            for component in a["rxs"][rx]:
                expected = a[f"rxs/{rx}/{component}"][:]
                actual = b[f"rxs/{rx}/{component}"][:]
                assert np.isfinite(expected).all() and np.isfinite(actual).all()
                scale = np.max(np.abs(expected))
                assert scale > 1e-8, (rx, component, "unexcited reference")
                if precision == "single":
                    # PML-wall probes are strongly attenuated. Scale device
                    # roundoff by the component's measured signal peak, not
                    # an individual attenuated trace. The same CPU/device
                    # residual exists at the unchanged lower PMC wall.
                    scale = max(np.max(np.abs(r[component][:])) for r in a["rxs"].values())
                error = float(np.max(np.abs(actual - expected)) / scale)
                maximum = max(maximum, error)
                assert error < (5e-5 if precision == "single" else 3e-11), (rx, component, error)
    return maximum


@pytest.mark.parametrize("stretch,wall", PAIRS)
@pytest.mark.parametrize(
    "formulation,order", [(f, n) for f in ("HORIPML", "MRIPML") for n in (1, 2)]
)
def test_upper_pmc_pml_matches_reflected_lower_wall(tmp_path, stretch, wall, formulation, order):
    lower, upper = tmp_path / "lower", tmp_path / "upper"
    for is_upper, output in ((False, lower), (True, upper)):
        _run(
            output, stretch=stretch, wall=wall, upper=is_upper, formulation=formulation, order=order
        )
    _compare(lower, upper)


@pytest.mark.parametrize("material", ("lossy", "debye", "lorentz"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
def test_dispersive_pmc_pml_mirror(tmp_path, material, formulation):
    lower, upper = tmp_path / "lower", tmp_path / "upper"
    for is_upper, output in ((False, lower), (True, upper)):
        _run(output, upper=is_upper, material=material, formulation=formulation)
    _compare(lower, upper)


@pytest.mark.parametrize(
    "backend",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ),
)
@pytest.mark.parametrize("stretch,wall", PAIRS)
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
def test_internal_terminal_plane_and_pmc_intersection(
    tmp_path, request, backend, stretch, wall, formulation
):
    options = {}
    if backend != "cpu":
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options["gpu" if backend == "cuda" else "opencl"] = [device]
    lower, upper = tmp_path / "lower", tmp_path / "upper"
    kwargs = dict(stretch=stretch, wall=wall, formulation=formulation, order=2, internal=True)
    _run(lower, upper=False, **kwargs)
    _run(upper, upper=True, options=options, **kwargs)
    _compare(lower, upper)


@pytest.mark.parametrize("omitted", ("electric", "magnetic", "both"))
def test_mirror_oracle_detects_either_missing_pml_correction(tmp_path, monkeypatch, omitted):
    from gprMax.pml import PML

    lower, upper = tmp_path / "lower", tmp_path / "upper"
    _run(lower, upper=False)
    for family in ("electric", "magnetic"):
        if omitted not in (family, "both"):
            continue
        original = getattr(PML, "update_" + family)

        def incomplete(slab, original=original):
            mask = slab.pmc_mask
            try:
                slab.pmc_mask = 0
                return original(slab)
            finally:
                slab.pmc_mask = mask

        monkeypatch.setattr(PML, "update_" + family, incomplete)
    _run(upper, upper=True)
    # Even a bounded waveform is wrong when only one family is corrected.
    with pytest.raises(AssertionError):
        _compare(lower, upper)


@pytest.mark.parametrize(
    "backend",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ),
)
def test_pmc_pml_histories_reset_with_reused_geometry(tmp_path, request, backend):
    options = {}
    if backend != "cpu":
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options["gpu" if backend == "cuda" else "opencl"] = [device]
    _run(tmp_path / "fresh", options=options, both=True)
    _run(tmp_path / "reused", options=dict(options, n=2, geometry_fixed=True), both=True)
    for index in (1, 2):
        assert _compare(tmp_path / "fresh", tmp_path / f"reused{index}") == 0


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
@pytest.mark.parametrize(
    "backend",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ),
)
def test_two_pmc_walls_with_pml_are_bounded_at_late_times(
    tmp_path, request, precision, formulation, backend
):
    output = tmp_path / "long"
    options = {}
    if backend != "cpu":
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options["gpu" if backend == "cuda" else "opencl"] = [device]
    _run(output, precision, options=options, both=True, steps=5000, formulation=formulation)
    with h5py.File(output.with_suffix(".h5")) as result:
        for rx in result["rxs"].values():
            for dataset in rx.values():
                values = dataset[:]
                assert np.isfinite(values).all()
                early_peak = np.max(np.abs(values[:1000]))
                assert early_peak > 1e-8
                assert np.max(np.abs(values[3000:])) < 0.02 * early_peak


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("backend", ("cuda", "opencl", "metal"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("stretch,wall", PAIRS)
@pytest.mark.parametrize(
    "formulation,order", [(f, n) for f in ("HORIPML", "MRIPML") for n in (1, 2)]
)
def test_device_pmc_pml_matches_cpu_mirror(
    tmp_path, request, backend, precision, stretch, wall, formulation, order
):
    if backend == "metal":
        if sys.platform != "darwin" or importlib.util.find_spec("Metal") is None:
            pytest.skip("requires Apple Metal")
        if precision == "double":
            pytest.skip("Metal uses single precision")
        options = {"metal": True}
    else:
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options = {"gpu" if backend == "cuda" else "opencl": [device]}
    reference, actual = tmp_path / "cpu_lower", tmp_path / backend
    scene_options = dict(stretch=stretch, wall=wall, formulation=formulation, order=order)
    _run(reference, precision, upper=False, **scene_options)
    _run(actual, precision, options=options, upper=True, **scene_options)
    _compare(reference, actual, precision)


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and mpiexec",
)
@pytest.mark.parametrize("partition", ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
def test_mpi_pmc_pml_matches_serial(tmp_path, partition, formulation):
    reference, actual = tmp_path / "cpu", tmp_path / "mpi"
    _run(reference, both=True, formulation=formulation, order=2)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
        MPI4PY_RC_INITIALIZE="1",
    )
    result = subprocess.run(
        [
            MPIEXEC,
            "-n",
            str(np.prod(partition)),
            sys.executable,
            str(Path(__file__).resolve()),
            str(actual),
            "--partition",
            *map(str, partition),
            "--formulation",
            formulation,
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _compare(reference, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--partition", nargs=3, type=int, required=True)
    parser.add_argument("--formulation", required=True)
    args = parser.parse_args()
    _run(
        args.output,
        options={"mpi": tuple(args.partition)},
        both=True,
        formulation=args.formulation,
        order=2,
    )
