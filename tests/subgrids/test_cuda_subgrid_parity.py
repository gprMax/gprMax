"""Real CUDA kernel/solve coverage for every CPU spline order and unity ratio."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.solvers import Solver
from gprMax.subgrids.precursor_nodes import PrecursorNodes, PrecursorNodesEqualResolution, PrecursorNodesFiltered
from gprMax.subgrids.subgrid_hsg import SubGridHSG

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
CASES = [(3, order, filtered) for order in range(1, 6) for filtered in (False, True)] + [
    (1, 1, True),
    (5, 3, True),
    (7, 5, False),
]


def scene(ratio=3, order=1, filtered=True):
    model = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.096, 0.090, 0.084)),
        gprMax.Discretisation(p1=(0.003,) * 3),
        gprMax.TimeWindow(iterations=100),
        gprMax.PMLThickness(thickness=4),
        gprMax.OMPThreads(1),
    ):
        model.add(obj)
    fine = gprMax.SubGridHSG(
        p1=(0.030,) * 3,
        p2=(0.063, 0.060, 0.057),
        ratio=ratio,
        id="fine",
        interpolation=order,
        filter=filtered,
    )
    model.add(fine)
    for owner, point, axis in ((model, (0.021, 0.036, 0.039), "z"), (fine, (0.045, 0.045, 0.042), "x")):
        owner.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=4e9, id="pulse"))
        owner.add(gprMax.HertzianDipole(p1=point, polarisation=axis, waveform_id="pulse"))
    model.add(gprMax.Rx(p1=(0.072, 0.039, 0.039), id="outer"))
    fine.add(gprMax.Rx(p1=(0.048, 0.048, 0.045), id="inner"))
    return model


def run(model, path, gpu=None, **kwargs):
    if gpu is not None:
        kwargs["gpu"] = [gpu]
    gprMax.run(
        scenes=[model],
        outputfile=path,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
        **kwargs,
    )


def compare_fields(expected, actual, *, rtol=1e-9):
    for name in FIELDS:
        peak = max(np.max(np.abs(expected[name[0] + axis])) for axis in "xyz")
        np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=rtol * max(peak, 1e-30), err_msg=name)


@pytest.mark.parametrize("ratio,order,filtered", CASES)
def test_cuda_random_precursors_and_interfaces_match_cpu(
    tmp_path,
    monkeypatch,
    gpu_device,
    ratio,
    order,
    filtered,
):
    class Finished(Exception):
        pass

    rng = np.random.default_rng(834)

    def probe(solver, iterator):
        parent, child = solver.updates, solver.updates.updaters[0]
        main, fine, device = parent.grid, child.grid, child.precursors
        cls = (
            PrecursorNodesEqualResolution
            if fine.equal_resolution
            else (PrecursorNodesFiltered if fine.filter else PrecursorNodes)
        )
        cpu = cls(main, fine)
        try:
            for _ in range(2):
                for name in FIELDS:
                    values = getattr(main, name)
                    values[:] = rng.normal(size=values.shape)
                    getattr(main, name + "_dev").set(values)
                for kind, names in (("electric", cpu.fn_e), ("magnetic", cpu.fn_m)):
                    getattr(cpu, "update_" + kind)()
                    getattr(device, "update_" + kind)()
                    for name in names:
                        np.testing.assert_allclose(
                            device.dev_1[name].get(), getattr(cpu, name + "_1"), rtol=1e-12, atol=1e-12, err_msg=name
                        )
                    for step in range(1, ratio + 1):
                        cpu.weight_pre_and_current_fields(step, names)
                        device.weight_pre_and_current_fields(step, names)
                        for name in names:
                            np.testing.assert_allclose(
                                device.dev[name].get(), getattr(cpu, name), rtol=1e-12, atol=1e-12, err_msg=name
                            )
                    cpu.calc_exact_field(names)
                    device.calc_exact_field(names)
            # Distinguish component, spatial and material-table indexing. A
            # uniform free-space ID array would hide a wrong lookup component.
            # This is an algebraic kernel probe, not a physical random medium.
            for grid in (main, fine):
                grid.ID[:] = rng.integers(0, 11, size=grid.ID.shape, dtype=grid.ID.dtype)
                grid.ID_dev.set(grid.ID)
                for kind in ("E", "H"):
                    name = "updatecoeffs" + kind
                    coefficients = getattr(grid, name)
                    setattr(grid, name, rng.uniform(0.05, 0.2, (11, coefficients.shape[1])).astype(coefficients.dtype))
                grid.htod_mat_coeff_arrays()
            for method in ("update_electric_is", "update_magnetic_is", "update_electric_os", "update_magnetic_os"):
                for grid in (main, fine):
                    for name in FIELDS:
                        values = getattr(grid, name)
                        values[:] = rng.normal(size=values.shape)
                        getattr(grid, name + "_dev").set(values)
                if method.endswith("is"):
                    getattr(SubGridHSG, method)(fine, cpu)
                    getattr(fine, method)(device)
                else:
                    getattr(SubGridHSG, method)(fine, main)
                    getattr(fine, method)(main)
                for grid in (main, fine):
                    for name in FIELDS:
                        np.testing.assert_allclose(
                            getattr(grid, name + "_dev").get(),
                            getattr(grid, name),
                            rtol=1e-11,
                            atol=1e-11,
                            err_msg=method + name,
                        )
        finally:
            parent.cleanup()
        raise Finished()

    monkeypatch.setattr(Solver, "solve", probe)
    with pytest.raises(Finished):
        run(scene(ratio, order, filtered), tmp_path / "probe", gpu_device)


@pytest.mark.parametrize("ratio,order,filtered", CASES)
def test_cuda_full_solve_and_reused_geometry_match_cpu(tmp_path, gpu_device, ratio, order, filtered):
    for backend in ("cpu", "cuda"):
        run(
            scene(ratio, order, filtered),
            tmp_path / backend,
            gpu_device if backend == "cuda" else None,
            n=2,
            geometry_fixed=True,
        )
    for index in (1, 2):
        with h5py.File(tmp_path / f"cpu{index}.h5") as cpu, h5py.File(tmp_path / f"cuda{index}.h5") as device:
            for prefix in ("", "subgrids/fine/"):
                expected = {name: cpu[prefix + "rxs/rx1/" + name][...] for name in FIELDS}
                actual = {name: device[prefix + "rxs/rx1/" + name][...] for name in FIELDS}
                compare_fields(expected, actual)
                assert any(np.max(np.abs(v)) > 1e-9 for v in expected.values())
    with h5py.File(tmp_path / "cuda1.h5") as first, h5py.File(tmp_path / "cuda2.h5") as second:
        for prefix in ("", "subgrids/fine/"):
            for name in FIELDS:
                np.testing.assert_array_equal(first[prefix + "rxs/rx1/" + name], second[prefix + "rxs/rx1/" + name])


def test_cuda_buffered_and_streamed_subgrid_snapshots_match_cpu(tmp_path, gpu_device, monkeypatch):
    from gprMax import config, contexts

    factory = contexts.create_solver
    observed = []
    for backend in ("cpu", "buffered", "streamed"):
        model = scene()
        fine = model.subgrid_objects[0]
        for owner, name, p1, p2, dl in (
            (model, "main", (0.012,) * 3, (0.075, 0.072, 0.069), (0.006,) * 3),
            (fine, "fine", (0.036,) * 3, (0.054, 0.054, 0.048), (0.002,) * 3),
        ):
            for step in (0, 17):
                owner.add(
                    gprMax.Snapshot(
                        p1=p1, p2=p2, dl=dl, iterations=step, filename=f"{name}_{step}", fileext=".h5", outputs=FIELDS
                    )
                )
        with monkeypatch.context() as patch:
            if backend != "cpu":

                def create_solver(model):
                    streaming = backend == "streamed"
                    # The normal memory policy may choose streaming; force
                    # each branch before any device snapshot buffer is made.
                    config.get_model_config().device["snapsgpu2cpu"] = streaming
                    solver = factory(model)
                    for updates in (solver.updates, *solver.updates.updaters):
                        expected = 1 if streaming else len(updates.grid.snapshots)
                        assert updates.snapEx_dev.shape[0] == expected
                    observed.append(streaming)
                    return solver

                patch.setattr(contexts, "create_solver", create_solver)
            run(model, tmp_path / backend, gpu_device if backend != "cpu" else None)
    assert observed == [False, True]
    references = sorted((tmp_path / "cpu_snaps").glob("*.h5"))
    assert len(references) == 4
    for reference in references:
        for backend in ("buffered", "streamed"):
            with h5py.File(reference) as cpu, h5py.File(tmp_path / f"{backend}_snaps" / reference.name) as device:
                assert cpu.attrs["iteration"] == device.attrs["iteration"]
                for component in FIELDS:
                    expected = cpu[component][...]
                    np.testing.assert_allclose(device[component][...], expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_unity_ratio_internal_pml_matches_cpu_and_monolithic_grid(tmp_path, gpu_device, formulation, precision):
    from tests.subgrids.test_equal_resolution_internal_pml import _scene, _histories

    histories = {}
    for backend, embedded in (("cpu", False), ("cpu", True), ("cuda", False), ("cuda", True)):
        key = backend + str(embedded)
        options = {} if backend == "cpu" else {"gpu": [gpu_device]}
        gprMax.run(
            scenes=[_scene(embedded, True, formulation)],
            outputfile=tmp_path / key,
            subgrid=embedded,
            autotranslate=embedded,
            cpu_precision=precision,
            gpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
            **options,
        )
        histories[key] = _histories(tmp_path / f"{key}.h5", embedded)
    for name, expected in histories["cpuFalse"].items():
        peak = np.max(np.abs(expected))
        assert peak > 1e-10
        tolerance = 2e-5 if precision == "single" else 1e-10
        for actual in histories.values():
            assert actual[name].dtype == expected.dtype
            np.testing.assert_allclose(actual[name], expected, rtol=tolerance, atol=tolerance * peak)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_unity_ratio_debye_currents_and_snapshots_match_monolithic_cpu(tmp_path, gpu_device, precision):
    from tests.subgrids.test_per_grid_dispersion import _parity_scene, _receiver

    gprMax.run(
        scenes=[_parity_scene(use_subgrid=False)],
        outputfile=tmp_path / "reference",
        cpu_precision=precision,
        hide_progress_bars=True,
        log_level=40,
    )
    tolerance = 5e-4 if precision == "single" else 1e-10
    for backend in ("cpu", "cuda"):
        options = {} if backend == "cpu" else {"gpu": [gpu_device]}
        gprMax.run(
            scenes=[_parity_scene(use_subgrid=True)],
            outputfile=tmp_path / backend,
            subgrid=True,
            autotranslate=True,
            cpu_precision=precision,
            gpu_precision=precision,
            n=2,
            geometry_fixed=True,
            hide_progress_bars=True,
            log_level=40,
            **options,
        )
        for index in (1, 2):
            with h5py.File(tmp_path / "reference.h5") as ref, h5py.File(tmp_path / f"{backend}{index}.h5") as actual:
                fine = actual["subgrids/local_grid"]
                assert fine.attrs["interpolation"] == 0
                assert not fine.attrs["filter"]
                assert fine.attrs["subgrid_pml_thickness"] == 0
                for name, owner, components in (("rx", actual, ("Ez", "Hy")), ("inside", fine, ("Ez", "Hy", "Iz"))):
                    a, b = _receiver(ref, name), _receiver(owner, name)
                    for component in components:
                        expected, observed = a[component][...], b[component][...]
                        assert np.max(np.abs(expected)) > 0
                        assert observed.dtype == expected.dtype
                        np.testing.assert_allclose(
                            observed, expected, rtol=tolerance, atol=tolerance * np.max(np.abs(expected))
                        )
            with h5py.File(tmp_path / "reference_snaps/parity_fields.h5") as ref, h5py.File(
                tmp_path / f"{backend}{index}_snaps/parity_fields.h5"
            ) as actual:
                assert actual.attrs["iteration"] == ref.attrs["iteration"] == 160
                for component in ("Ez", "Hy"):
                    expected = ref[component][...]
                    np.testing.assert_allclose(
                        actual[component][...], expected, rtol=tolerance, atol=tolerance * np.max(np.abs(expected))
                    )
