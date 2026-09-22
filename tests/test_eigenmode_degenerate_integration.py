"""End-to-end channel labels and physical/virtual guide equivalence."""
from unittest.mock import patch

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.grid.fdtd_grid import FDTDGrid
from testing.validation.degenerate_eigenmode_ports import circular_scene


def run(path, *, randomize=False, precision="double", device_options=None, **kwargs):
    original = FDFD_2D_mode_solver.solve
    rng = np.random.default_rng(62)
    grids = []
    build = FDTDGrid.build

    def capture(grid):
        build(grid)
        for monitor in grid.eigenmodeports:
            port = monitor.owner
            masks = port._cell_pec_electric_component_masks(grid)
            tensors = port._extract_local_complex_property_tensors(grid, electric=True)
            for mask, values in zip(masks, tensors):
                assert not np.any(mask & np.isfinite(values))
        grids.append(grid)

    def solve(solver):
        original(solver)
        if randomize:
            rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
                fields = getattr(solver, name)
                fields[..., :2] = fields[..., :2] @ rotation

    with patch.object(FDFD_2D_mode_solver, "solve", solve), patch.object(
        FDTDGrid, "build", capture
    ):
        gprMax.run(
            scenes=[circular_scene(**kwargs)],
            outputfile=path,
            cpu_precision=precision,
            gpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
            **(device_options or {}),
        )
    with h5py.File(path.with_suffix(".h5")) as output:
        traces = np.array(
            [
                [
                    output[f"rxs/rx{rx}/{field}"][...]
                    for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                ]
                for rx in (1, 2)
            ]
        )
        metadata = output["eigenmode_ports/port1/degenerate_groups/1"]
        assert tuple(metadata.attrs["ModeIndices"]) == (1, 2)
        assert metadata.attrs["PhysicalPolarization"]
        assert np.max(metadata["anchor0/residual"]) < 1e-9
    return traces, grids[0]


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis,direction", [(a, d) for a in range(3) for d in ("+", "-")])
@pytest.mark.parametrize("precision", ("double", "single"))
@pytest.mark.parametrize("mode,combination", [(1, None), (2, None), (1, 0), (1, 90)])
def test_physical_virtual_and_random_basis(
    tmp_path, normal_axis, direction, precision, mode, combination, record_property
):
    options = dict(
        normal_axis=normal_axis,
        direction=direction,
        precision=precision,
        mode=mode,
        combination=combination,
    )
    reference, _ = run(tmp_path / "physical", virtual=False, **options)
    actual, grid = run(tmp_path / "virtual", virtual=True, randomize=True, **options)
    # E and eta0 H have the same units; do not amplify roundoff-only components.
    weights = np.array([1, 1, 1, 376.730313668, 376.730313668, 376.730313668])[None, :, None]
    error = np.max(abs((actual - reference) * weights)) / np.max(abs(reference * weights))
    record_property("normalized_field_error", float(error))
    assert error < (2e-8 if precision == "double" else 2e-4), error
    assert np.max(abs(reference)) > 0
    guide = grid.virtual_waveguides[0]
    guide.reset_run_state()
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert not np.any(getattr(guide.aux_grid, name))


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis,direction", [(a, d) for a in range(3) for d in ("+", "-")])
@pytest.mark.parametrize("precision", ("double", "single"))
def test_broadband_random_basis(tmp_path, normal_axis, direction, precision, record_property):
    options = dict(
        broadband=True,
        steps=1200,
        monitor=True,
        combination=90,
        normal_axis=normal_axis,
        direction=direction,
        precision=precision,
    )
    reference, _ = run(tmp_path / "reference", virtual=False, **options)
    actual, grid = run(tmp_path / "random", virtual=True, randomize=True, **options)
    weights = np.array([1, 1, 1, 376.730313668, 376.730313668, 376.730313668])[None, :, None]
    error = np.max(abs((actual - reference) * weights)) / np.max(abs(reference * weights))
    record_property("normalized_field_error", float(error))
    assert error < (2e-8 if precision == "double" else 2e-4)
    assert len(grid.virtual_waveguides[0].port.port_anchor_frequencies) > 1
    assert (
        grid.eigenmodereceivers[0].mode_polarizations
        == grid.virtual_waveguides[0].port.mode_polarizations
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("normal_axis,direction", [(a, d) for a in range(3) for d in ("+", "-")])
@pytest.mark.parametrize("precision", ("double", "single"))
@pytest.mark.parametrize("mode,broadband", [(1, False), (2, False), (1, True)])
def test_device_physical_labels_and_virtual_continuation(
    tmp_path, request, backend, normal_axis, direction, precision, mode, broadband, record_property
):
    """Aligned channels must reach device injection, including a mixed pulse."""
    fixture = "gpu_device" if backend == "cuda" else "opencl_device"
    device_options = {"gpu" if backend == "cuda" else "opencl": [request.getfixturevalue(fixture)]}
    options = dict(
        normal_axis=normal_axis,
        direction=direction,
        precision=precision,
        mode=mode,
        broadband=broadband,
        combination=90 if broadband else None,
        steps=1200 if broadband else 400,
        monitor=broadband,
    )
    reference, _ = run(tmp_path / "cpu", virtual=False, **options)
    weights = np.array([1, 1, 1, 376.730313668, 376.730313668, 376.730313668])[None, :, None]
    scale = np.max(abs(reference * weights))
    assert scale > 0
    for virtual in (False, True):
        actual, _ = run(
            tmp_path / ("virtual" if virtual else "physical"),
            virtual=virtual,
            randomize=True,
            device_options=device_options,
            **options,
        )
        error = np.max(abs((actual - reference) * weights)) / scale
        record_property("virtual_error" if virtual else "physical_error", float(error))
        assert error < (2e-8 if precision == "double" else 2e-4), error


@pytest.mark.integration
def test_cached_modal_study_preserves_physical_labels(tmp_path, monkeypatch):
    original = FDFD_2D_mode_solver.solve
    calls = []
    rng = np.random.default_rng(103)

    def solve(solver):
        original(solver)
        calls.append(1)
        rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
            fields = getattr(solver, name)
            fields[..., :2] = fields[..., :2] @ rotation

    monkeypatch.setattr(FDFD_2D_mode_solver, "solve", solve)
    scene = circular_scene()
    excitation = next(
        obj for obj in scene.grid_objects if isinstance(obj, gprMax.EigenmodeExcitation)
    )
    study = gprMax.EigenmodeStudy(
        [
            gprMax.StudyCase(f"mode{mode}", [gprMax.ObjectState(excitation, port=1, mode=mode)])
            for mode in (1, 2)
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "study",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    assert len(calls) == 1
    for mode in (1, 2):
        fresh, _ = run(tmp_path / f"fresh{mode}", mode=mode)
        with h5py.File(tmp_path / f"study{mode}.h5") as output:
            reused = np.array(
                [
                    [
                        output[f"rxs/rx{rx}/{field}"][...]
                        for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                    ]
                    for rx in (1, 2)
                ]
            )
            assert output["study/eigenmode_response"].attrs["InputMode"] == mode
            assert "degenerate_groups" in output["eigenmode_ports/port1"]
        assert np.max(abs(reused - fresh)) / np.max(abs(fresh)) < 2e-8


@pytest.mark.integration
@pytest.mark.parametrize("syntax", ("python", "hash"))
def test_geometry_export_and_plots_use_physical_bank(tmp_path, monkeypatch, syntax):
    from matplotlib.figure import Figure

    import gprMax.eigenmode_plotting as plotting

    vectors, titles = [], []
    plot = plotting._plot_2d_vector

    def capture_vector(axes, solver, u, v, phase, family):
        vectors.append((u.copy(), v.copy(), phase, family))
        return plot(axes, solver, u, v, phase, family)

    def capture_figure(figure, *args, **kwargs):
        titles.append(figure._suptitle.get_text())

    monkeypatch.setattr(plotting, "_plot_2d_vector", capture_vector)
    monkeypatch.setattr(Figure, "savefig", capture_figure)
    scene = circular_scene()
    port = next(obj for obj in scene.grid_objects if isinstance(obj, gprMax.EigenmodePort))
    port.kwargs["plot_fields"] = True
    if syntax == "python":
        scene.add(gprMax.EigenmodeFieldOutput(filename="bank"))
    else:
        from gprMax.hash_cmds_file import get_user_objects

        (output,) = get_user_objects(["#eigenmode_field_output: bank 1\n"], checkessential=False)
        scene.add(output)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "geometry",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    assert len(titles) == 2 and all("E direction" in title for title in titles)
    assert len(vectors) == 4 and all(item[2] == 1 for item in vectors)
    with h5py.File(tmp_path / "bank.modes.h5") as output:
        port = output["ports/1"]
        np.testing.assert_array_equal(
            port["degenerate_groups/1/requested_directions"], [[0, 1, 0], [1, 0, 0]]
        )
        for i, (u, v, _, family) in enumerate(vectors):
            mode = i // 2
            np.testing.assert_allclose(port[f"fields/{family}x/values"][0, mode], u, atol=1e-10)
            np.testing.assert_allclose(port[f"fields/{family}y/values"][0, mode], v, atol=1e-10)
