"""CUDA/CPU virtual-guide parity on main/fine grids and across reused runs."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.subgrids import cuda_subgrid_updates as hsg
from gprMax.subgrids.cuda_precursor_nodes import CUDAPrecursorNodesFiltered
from gprMax.subgrids.cuda_subgrid_hsg import CUDASubGridHSG
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.virtual_waveguide import VirtualWaveguide
from tests.cmds_multiuse.test_eigenmode_subgrid import _add_modal_commands, _subgrid_scene

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _scene(placement, normal_axis=0, *, virtual=True, ratio=3):
    if placement == "fine":
        scene, owner = _subgrid_scene(normal_axis=normal_axis, virtual_waveguide=virtual)
        owner.kwargs["ratio"] = ratio
        if ratio == 1:
            # Retain the guide's resolved 1 mm lattice without refinement.
            for index, obj in enumerate(scene.single_use_objects):
                if isinstance(obj, gprMax.Discretisation):
                    scene.single_use_objects[index] = gprMax.Discretisation(p1=(0.001,) * 3)
        offset = np.full(3, 0.03)
        prefix = "subgrids/fine_grid/"
    else:
        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=(0.06,) * 3),
            gprMax.Discretisation(p1=(0.001,) * 3),
            gprMax.TimeWindow(time=5e-10),
            gprMax.PMLThickness(thickness=0),
            gprMax.OMPThreads(1),
        ):
            scene.add(obj)
        owner, offset, prefix = scene, np.zeros(3), ""
        _add_modal_commands(
            owner,
            offset=offset,
            normal_axis=normal_axis,
            passive_port=True,
            virtual_waveguide=virtual,
        )
        scene.add(gprMax.SubGridHSG(p1=(0.036,) * 3, p2=(0.048,) * 3, ratio=ratio, id="fine"))
    if virtual:
        # Both directions: port 1 launches into the model; port 2 is a matched
        # passive guide until the study switches the excitation to it.
        owner.add(
            gprMax.VirtualWaveguide(
                port=2,
                length_cells=12,
                pml_cells=4,
                source_clearance_cells=3,
            )
        )
    point = np.full(3, 0.015)
    point[normal_axis] = 0.021
    owner.add(gprMax.Rx(p1=tuple(point + offset), id="probe"))
    return scene, owner, prefix


def _run(scene, output, gpu_device=None, *, precision="double", **options):
    if gpu_device is not None:
        options["gpu"] = [gpu_device]
    gprMax.run(
        scenes=[scene],
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        cpu_precision=precision,
        gpu_precision=precision,
        hide_progress_bars=True,
        log_level=30,
        **options,
    )


def _read(path, prefix):
    with h5py.File(path) as output:
        group = output[prefix] if prefix else output
        result = {
            f"port{port}/{name}": group[f"eigenmode_ports/port{port}/{name}"][...]
            for port in (1, 2)
            for name in ("incident", "outgoing", "S")
        }
        result.update({name: values[...] for name, values in group["rxs/rx1"].items()})
        return result


def _compare(reference, actual, *, rtol=1e-8, atol_scale=1e-10):
    assert reference.keys() == actual.keys()
    assert np.max(np.abs(reference["port1/incident"])) > 1e-12
    for name, values in reference.items():
        assert np.isfinite(actual[name]).all(), name
        peak = np.max(np.abs(values))
        if name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            # A symmetry-null component is compared to its vector's scale.
            peak = max(np.max(np.abs(reference[name[0] + axis])) for axis in "xyz")
        np.testing.assert_allclose(
            actual[name],
            values,
            rtol=rtol,
            atol=atol_scale * max(peak, 1e-20),
            err_msg=name,
        )


@pytest.mark.parametrize("placement", ["main", "fine"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_unity_virtual_guides_match_cpu(tmp_path, gpu_device, placement, precision):
    results = []
    for backend in ("cpu", "cuda"):
        scene, _, prefix = _scene(placement, ratio=1)
        _run(scene, tmp_path / backend, gpu_device if backend == "cuda" else None, precision=precision)
        results.append(_read(tmp_path / f"{backend}.h5", prefix))
    dtype = np.float32 if precision == "single" else np.float64
    assert results[0]["Ex"].dtype == results[1]["Ex"].dtype == dtype
    tolerance = 5e-5 if precision == "single" else 1e-10
    _compare(*results, rtol=tolerance, atol_scale=tolerance)


@pytest.mark.parametrize("placement", ["main", "fine"])
@pytest.mark.parametrize("normal_axis", range(3), ids=["x", "y", "z"])
def test_cuda_virtual_guides_match_cpu_across_repeated_runs(
    tmp_path,
    monkeypatch,
    gpu_device,
    placement,
    normal_axis,
):
    calls = []
    original = VirtualWaveguide.initialise_device

    def initialise(guide, parent):
        original(guide, parent)
        assert type(guide.aux_updates) is CUDAUpdates
        assert guide.aux_updates.ctx is parent.ctx
        assert not guide.aux_updates._owns_context
        assert guide.aux_grid.dt == parent.grid.dt
        assert guide.aux_grid.iterations == parent.grid.iterations
        calls.append(guide.aux_grid.dt)

    monkeypatch.setattr(VirtualWaveguide, "initialise_device", initialise)
    for backend in ("cpu", "cuda"):
        for index in (1, 2):
            scene, _, prefix = _scene(placement, normal_axis)
            # Raw geometry_fixed is deliberately forbidden for modal sources;
            # reuse is covered through the supported EigenmodeStudy below.
            _run(scene, tmp_path / f"{backend}{index}", gpu_device if backend == "cuda" else None)
    assert len(calls) == 4  # active and passive auxiliaries constructed on each run
    first = _read(tmp_path / "cuda1.h5", prefix)
    for index in (1, 2):
        cpu = _read(tmp_path / f"cpu{index}.h5", prefix)
        device = _read(tmp_path / f"cuda{index}.h5", prefix)
        _compare(cpu, device)
        for name in device:
            np.testing.assert_array_equal(device[name], first[name], err_msg=name)


@pytest.mark.parametrize("placement,virtual", [("main", True), ("fine", True), ("fine", False)])
def test_cuda_modal_study_matches_cpu_when_switching_active_port(
    tmp_path,
    gpu_device,
    placement,
    virtual,
):
    results = []
    for backend in ("cpu", "cuda"):
        scene, owner, prefix = _scene(placement, virtual=virtual)
        objects = owner.grid_objects if placement == "main" else owner.children_grid
        excitation = next(obj for obj in objects if isinstance(obj, gprMax.EigenmodeExcitation))
        study = gprMax.EigenmodeStudy(
            [gprMax.StudyCase(f"drive_{port}", [gprMax.ObjectState(excitation, port=port, mode=1)]) for port in (1, 2)]
        )
        _run(scene, tmp_path / backend, gpu_device if backend == "cuda" else None, study=study)
        results.append(study.result)
    assert results[0].coefficient_valid_s.any()
    np.testing.assert_array_equal(results[1].coefficient_valid_s, results[0].coefficient_valid_s)
    np.testing.assert_allclose(results[1].s, results[0].s, rtol=1e-8, atol=1e-10)
    for index in (1, 2):
        _compare(_read(tmp_path / f"cpu{index}.h5", prefix), _read(tmp_path / f"cuda{index}.h5", prefix))


@pytest.mark.parametrize(
    "stage", ["main_guide", "fine_guide", "coefficients", "child_kernels", "interface", "precursors"]
)
def test_cuda_subgrid_failure_releases_context_and_allows_another_model(
    tmp_path,
    monkeypatch,
    gpu_device,
    stage,
):
    import pycuda.driver as cuda

    before = cuda.Context.get_current()
    placement = "main" if stage == "main_guide" else "fine"
    target, method = {
        "main_guide": (VirtualWaveguide, "initialise_device"),
        "fine_guide": (VirtualWaveguide, "initialise_device"),
        "coefficients": (hsg, "_upload_mat_coeffs"),
        "child_kernels": (hsg.CUDASubgridUpdater, "_set_subgrid_knls"),
        "interface": (CUDASubGridHSG, "setup_cuda_interface"),
        "precursors": (CUDAPrecursorNodesFiltered, "setup_device"),
    }[stage]

    def fail(*args, **kwargs):
        raise RuntimeError("controlled HSG setup failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(target, method, fail)
            scene, _, _ = _scene(placement)
            with pytest.raises(RuntimeError, match="controlled HSG setup failure"):
                _run(scene, tmp_path / "failed", gpu_device)
        assert cuda.Context.get_current() == before
        scene, _, prefix = _scene(placement)
        _run(scene, tmp_path / "recovery", gpu_device)
        assert cuda.Context.get_current() == before
        assert np.isfinite(_read(tmp_path / "recovery.h5", prefix)["port1/incident"]).all()
    finally:
        # Keep a future regression from aborting the pytest process at exit.
        current = cuda.Context.get_current()
        if current is not None and current != before:
            current.pop()
            current.detach()
