"""Singleton Scene repetition must use the existing rebuild/stepping paths."""

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
from gprMax import config

pytestmark = pytest.mark.integration


def make_scene(stepping="both", amplitude=1):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.040, 0.024, 0.024)),
        gprMax.Discretisation(p1=(0.002,) * 3),
        gprMax.TimeWindow(iterations=100),
        gprMax.PMLThickness(thickness=2),
        gprMax.OMPThreads(n=1),
        gprMax.Waveform(wave_type="gaussian", amp=amplitude, freq=5e9, id="pulse"),
        gprMax.HertzianDipole(p1=(0.008, 0.012, 0.012), polarisation="z", waveform_id="pulse"),
        gprMax.Rx(p1=(0.014, 0.012, 0.012), outputs=["Ez"], id="measurement"),
    ):
        scene.add(obj)
    if stepping == "both":
        scene.add(gprMax.SrcSteps(p1=(0.002, 0, 0)))
    if stepping != "none":
        scene.add(gprMax.RxSteps(p1=(0.002, 0, 0)))
    return scene


def run(path, scenes, **kwargs):
    return gprMax.run(
        scenes=scenes,
        outputfile=path,
        cpu_precision="double",
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
        **kwargs,
    )


def compare_runs(expected, actual, start, count, stepping):
    for index in range(start, start + count):
        with h5py.File(f"{expected}{index}.h5") as a, h5py.File(f"{actual}{index}.h5") as b:
            values = a["rxs/rx1/Ez"][:]
            assert np.isfinite(values).all() and np.any(values)
            np.testing.assert_array_equal(values, b["rxs/rx1/Ez"][:])
            for key in ("Iterations", "dt", "dx_dy_dz", "nx_ny_nz"):
                np.testing.assert_array_equal(a.attrs[key], b.attrs[key])
            for group, origin, moving in (
                ("rxs/rx1", 0.014, stepping != "none"),
                ("srcs/src1", 0.008, stepping == "both"),
            ):
                position = a[group].attrs["Position"]
                np.testing.assert_array_equal(position, b[group].attrs["Position"])
                np.testing.assert_allclose(
                    position, [origin + (index - 1) * 0.002 * moving, 0.012, 0.012]
                )


@pytest.mark.parametrize("start", [1, 4])
@pytest.mark.parametrize("stepping", ["none", "receiver", "both"])
def test_singleton_matches_explicit_repetition_and_geometry_reuse(
    tmp_path, monkeypatch, start, stepping
):
    scene = make_scene(stepping)
    builds = []
    original = gprMax.Scene.create_internal_objects

    def observe(self, model):
        builds.append(config.sim_config.current_model)
        return original(self, model)

    monkeypatch.setattr(gprMax.Scene, "create_internal_objects", observe)
    run(tmp_path / "single", [scene], n=3, i=start)
    assert builds == list(range(start - 1, start + 2))
    builds.clear()
    run(tmp_path / "explicit", [scene] * 3, n=3, i=start)
    assert builds == list(range(start - 1, start + 2))
    builds.clear()
    run(tmp_path / "fixed", [scene], n=3, i=start, geometry_fixed=True)
    assert builds == [start - 1]
    compare_runs(tmp_path / "single", tmp_path / "explicit", start, 3, stepping)
    compare_runs(tmp_path / "single", tmp_path / "fixed", start, 3, stepping)


def test_distinct_scenes_still_change_the_model(tmp_path):
    run(tmp_path / "distinct", [make_scene("none"), make_scene("none", 2)], n=2)
    with h5py.File(tmp_path / "distinct1.h5") as a, h5py.File(tmp_path / "distinct2.h5") as b:
        np.testing.assert_array_equal(2 * a["rxs/rx1/Ez"][:], b["rxs/rx1/Ez"][:])


def test_time_based_snapshot_is_not_reinterpreted_on_rebuild(tmp_path, capsys):
    """Ensure time-based snapshots do not trigger a false time-and-iterations
    warning when scenes are rebuilt."""
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.TimeWindow(iterations=20))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=5e9, id="w"))
    scene.add(
        gprMax.Snapshot(
            p1=(0.0, 0.0, 0.0),
            p2=(0.02, 0.02, 0.02),
            dl=(0.002,) * 3,
            filename="snap",
            time=1.5e-11,
        )
    )
    capsys.readouterr()
    gprMax.run(
        scenes=[scene],
        n=3,
        outputfile=tmp_path / "snapshot_rebuild",
        hide_progress_bars=True,
    )
    assert "Time and iterations were both specified" not in capsys.readouterr().out


@pytest.mark.parametrize("empty", [False, True])
def test_invalid_scene_counts_produce_no_partial_outputs(tmp_path, monkeypatch, empty):
    def unexpected_build(*args):
        pytest.fail("Invalid Scene counts must be rejected before building")

    monkeypatch.setattr(gprMax.Scene, "create_internal_objects", unexpected_build)
    with pytest.raises(ValueError, match="at least one Scene|only 2 Scene entries"):
        run(tmp_path / "invalid", [] if empty else [make_scene(), make_scene()], n=3)
    assert not list(tmp_path.glob("*.h5"))


def test_hash_input_is_reprocessed_for_each_run(tmp_path):
    infile = tmp_path / "dynamic.in"
    infile.write_text(
        "#domain: 0.040 0.024 0.024\n#dx_dy_dz: 0.002 0.002 0.002\n"
        "#time_window: 100\n#pml_cells: 2\n#num_threads: 1\n"
        "#python:\nprint('#waveform: gaussian', current_model_run, '5e9 pulse')\n#end_python:\n"
        "#hertzian_dipole: z 0.008 0.012 0.012 pulse\n#rx: 0.014 0.012 0.012\n"
    )
    gprMax.run(
        inputfile=infile,
        n=3,
        i=4,
        outputfile=tmp_path / "hash",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    run(tmp_path / "reference", [make_scene("none", amplitude=i) for i in (4, 5, 6)], n=3, i=4)
    compare_runs(tmp_path / "reference", tmp_path / "hash", 4, 3, "none")


@pytest.mark.gpu
def test_cuda_singleton_rebuild_and_reuse(tmp_path, gpu_device):
    scene = make_scene()
    run(tmp_path / "single", [scene], n=3, i=4, gpu=[gpu_device])
    run(tmp_path / "explicit", [scene] * 3, n=3, i=4, gpu=[gpu_device])
    run(tmp_path / "fixed", [scene], n=3, i=4, gpu=[gpu_device], geometry_fixed=True)
    compare_runs(tmp_path / "single", tmp_path / "explicit", 4, 3, "both")
    compare_runs(tmp_path / "single", tmp_path / "fixed", 4, 3, "both")


@pytest.mark.parametrize("mode", ["domain", "taskfarm"])
def test_mpi_singleton_matches_explicit_and_serial(tmp_path, mode):
    launcher = Path(sys.executable).with_name("mpiexec.exe" if os.name == "nt" else "mpiexec")
    executable = str(launcher) if launcher.is_file() else shutil.which("mpiexec")
    if executable is None or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("requires mpi4py and an MPI launcher")
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    environment.update(
        PYTHONPATH=str(root) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        executable,
        "-n",
        "2" if mode == "domain" else "3",
        sys.executable,
        str(Path(__file__).resolve()),
        str(tmp_path),
        mode,
    ]
    try:
        result = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"MPI Scene repetition did not finish within 60 s: {exc.stdout}\n{exc.stderr}")
    assert result.returncode == 0, result.stdout + result.stderr
    run(tmp_path / "serial", [make_scene()], n=3, i=5)
    compare_runs(tmp_path / "single", tmp_path / "explicit", 5, 3, "both")
    compare_runs(tmp_path / "serial", tmp_path / "single", 5, 3, "both")
    if mode == "domain":
        compare_runs(tmp_path / "single", tmp_path / "fixed", 5, 3, "both")


if __name__ == "__main__":
    directory, mode = Path(sys.argv[1]), sys.argv[2]
    options = {"mpi": [2, 1, 1]} if mode == "domain" else {"taskfarm": True}
    scene = make_scene()
    # The last case crosses the x partition, checking the absolute stepping
    # index as well as singleton lookup on every rank/worker.
    run(directory / "single", [scene], n=3, i=5, **options)
    run(directory / "explicit", [scene] * 3, n=3, i=5, **options)
    if mode == "domain":
        run(directory / "fixed", [scene], n=3, i=5, geometry_fixed=True, **options)
