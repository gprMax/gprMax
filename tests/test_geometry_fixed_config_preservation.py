"""Regression tests for a severe geometry_fixed bug (external review finding,
2026-07-21): every model run creates a brand-new ModelConfig
(Context._run_model()/MPIContext._run_model(), gprMax/contexts.py), with
all its defaults (mode="3D", materials["maxpoles"]=0, ...). On a
geometry_fixed reuse run, Model.build() takes the reuse_geometry() path
(Model.reuse_geometry(), gprMax/model.py) instead of build_geometry() -
which is where Domain.build() sets `mode` and
_check_for_dispersive_materials() sets materials["maxpoles"/
"drudelorentz"/"dispersivedtype"/"dispersiveCdtype"/"crealfunc"] - so
none of that ever gets (re)established for run 2 onward. Only
`ompthreads` was being restored (a prior session's fix for a crash, not
this deeper silent-correctness issue).

Confirmed concrete consequences before the fix:
- run 2+ reported mode "3D" regardless of the model's real mode, so code
  inspecting the current configuration no longer agreed with the reused
  grid geometry.
- FDTDGrid.reset_fields() checks materials["maxpoles"] > 0 before calling
  initialise_dispersive_arrays() - run 2+ would see maxpoles=0 for a
  genuinely dispersive model, skipping Tx/Ty/Tz reinitialisation and
  leaking the previous run's polarisation-current state into the next.
- gprMax/updates/cuda_updates.py (and OpenCL/Metal) read
  materials["maxpoles"] in _set_field_knls()/_set_src_knls() to decide
  whether to build/upload dispersive kernels/arrays at all.
Fixed by ModelConfig.restore_geometry_derived_config() (gprMax/config.py),
copying mode and materials from the ModelConfig that actually built the geometry
(sim_config.get_model_config(sim_config.model_start)) onto every reused
run's fresh ModelConfig, called from both Context._run_model() and
MPIContext._run_model().
"""
import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.updates.cpu_updates import CPUUpdates

try:
    import pycuda.autoinit  # noqa: F401

    HAS_CUDA = True
except Exception:
    HAS_CUDA = False


def _capture_modes(monkeypatch):
    captured = []
    orig_init = CPUUpdates.__init__

    def patched_init(self, grid):
        orig_init(self, grid)
        captured.append(config.get_model_config().mode)

    monkeypatch.setattr(CPUUpdates, "__init__", patched_init)
    return captured


def _capture_materials(monkeypatch):
    captured = []
    orig_init = CPUUpdates.__init__

    def patched_init(self, grid):
        orig_init(self, grid)
        captured.append(dict(config.get_model_config().materials))

    monkeypatch.setattr(CPUUpdates, "__init__", patched_init)
    return captured


def _base_2d_tm_scene(dl=1e-3, domain_transverse=0.02):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(domain_transverse, domain_transverse, dl)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z", p1=(0.01, 0.01, 0), waveform_id="w"
        )
    )
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0)))
    return scene


def test_2d_tm_mode_preserved_across_geometry_fixed_runs(monkeypatch, tmp_path):
    captured = _capture_modes(monkeypatch)
    scene = _base_2d_tm_scene()

    gprMax.run(
        scenes=[scene], n=3, geometry_fixed=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    assert captured == ["2D TMz", "2D TMz", "2D TMz"]


def _dispersive_scene(material_type, dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=3e-10))
    # Finite (not infinite) conductivity, ordinary dielectric - unlike an
    # infinite-conductivity/PEC-like material (see
    # tests/materials/test_dispersive_infinite_conductivity.py), the
    # dispersive correction genuinely, measurably affects the field here,
    # which matters for the "identical traces" leaked-state check below:
    # with se=inf the dispersive term has no visible effect on Ez at all,
    # making that check pass regardless of whether the bug exists.
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="mymat"))
    if material_type == "debye":
        scene.add(
            gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-10], material_ids=["mymat"])
        )
    else:
        scene.add(
            gprMax.AddLorentzDispersion(
                poles=1, er_delta=[2.0], omega=[2e10], delta=[5e9], material_ids=["mymat"]
            )
        )
    scene.add(gprMax.Box(p1=(0.008, 0.008, 0.008), p2=(0.012, 0.012, 0.012), material_id="mymat"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0.01)))
    return scene


@pytest.mark.parametrize("material_type,expected_drudelorentz", [("debye", False), ("lorentz", True)])
def test_dispersive_materials_config_preserved_across_geometry_fixed_runs(
    monkeypatch, tmp_path, material_type, expected_drudelorentz
):
    captured = _capture_materials(monkeypatch)
    scene = _dispersive_scene(material_type)

    gprMax.run(
        scenes=[scene], n=3, geometry_fixed=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    assert len(captured) == 3
    for materials in captured:
        assert materials["maxpoles"] == 1
        assert materials["drudelorentz"] is expected_drudelorentz


@pytest.mark.parametrize("material_type", ["debye", "lorentz"])
def test_dispersive_geometry_fixed_runs_produce_identical_traces_not_leaked_state(
    tmp_path, material_type
):
    """The real-world consequence check: with no source/receiver stepping,
    every geometry_fixed run uses an IDENTICAL setup, so the receiver
    trace must be identical run-to-run. Before the fix, run 2+ would
    silently skip Tx/Ty/Tz reinitialisation (materials["maxpoles"] wrongly
    read as 0), leaking run 1's final polarisation-current state into
    run 2 - producing DIFFERENT traces despite identical setup."""
    import h5py

    scene = _dispersive_scene(material_type)
    outputfile = tmp_path / "run"
    gprMax.run(
        scenes=[scene], n=3, geometry_fixed=True,
        outputfile=outputfile, hide_progress_bars=True,
    )

    traces = []
    for i in (1, 2, 3):
        with h5py.File(f"{outputfile}{i}.h5", "r") as f:
            ez = f["rxs/rx1/Ez"][:]
            assert not np.any(np.isnan(ez))
            assert np.max(np.abs(ez)) > 0  # sanity: genuinely propagating field
            traces.append(ez)

    assert np.array_equal(traces[0], traces[1])
    assert np.array_equal(traces[0], traces[2])


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA/pycuda available in this environment")
def test_dispersive_geometry_fixed_gpu_matches_cpu(tmp_path):
    """GPU configuration path: CUDAUpdates._set_field_knls() reads
    materials["maxpoles"] to decide whether to build dispersive kernels
    at all - verify a geometry_fixed dispersive run on real GPU hardware
    still gets dispersive kernels on run 2+ (matching CPU results, not
    silently falling back to non-dispersive updates)."""
    import h5py

    outputfile_cpu = tmp_path / "cpu"
    gprMax.run(
        scenes=[_dispersive_scene("debye")], n=3, geometry_fixed=True,
        outputfile=outputfile_cpu, hide_progress_bars=True,
    )
    outputfile_gpu = tmp_path / "gpu"
    gprMax.run(
        scenes=[_dispersive_scene("debye")], n=3, geometry_fixed=True, gpu=[0],
        outputfile=outputfile_gpu, hide_progress_bars=True,
    )

    for i in (1, 2, 3):
        with h5py.File(f"{outputfile_cpu}{i}.h5", "r") as fc, h5py.File(
            f"{outputfile_gpu}{i}.h5", "r"
        ) as fg:
            ez_cpu = fc["rxs/rx1/Ez"][:]
            ez_gpu = fg["rxs/rx1/Ez"][:]
            assert not np.any(np.isnan(ez_gpu))
            maxabs = np.max(np.abs(ez_cpu))
            assert maxabs > 0
            reldiff = np.max(np.abs(ez_cpu - ez_gpu)) / maxabs
            assert reldiff < 1e-4, f"run {i}: relative diff {reldiff} too large"


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA/pycuda available in this environment")
def test_dispersive_geometry_fixed_gpu_runs_produce_identical_traces_not_leaked_state(tmp_path):
    """GPU-side equivalent of test_dispersive_geometry_fixed_runs_produce_
    identical_traces_not_leaked_state - CPU-vs-GPU agreement alone
    wouldn't catch this bug if it affected both backends identically
    (materials["maxpoles"] is read from the same singleton ModelConfig by
    both CUDAUpdates and CPUUpdates), so this checks GPU run-to-run
    consistency directly, independent of any CPU comparison."""
    import h5py

    outputfile = tmp_path / "gpu_run"
    gprMax.run(
        scenes=[_dispersive_scene("debye")], n=3, geometry_fixed=True, gpu=[0],
        outputfile=outputfile, hide_progress_bars=True,
    )

    traces = []
    for i in (1, 2, 3):
        with h5py.File(f"{outputfile}{i}.h5", "r") as f:
            ez = f["rxs/rx1/Ez"][:]
            assert not np.any(np.isnan(ez))
            assert np.max(np.abs(ez)) > 0
            traces.append(ez)

    assert np.array_equal(traces[0], traces[1])
    assert np.array_equal(traces[0], traces[2])
