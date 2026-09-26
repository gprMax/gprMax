import numpy as np
import pytest

from testing.validation.impedance_surface.validate_2d import (
    FIELDS,
    compare_invariant_3d,
    normalized_field_error,
    run_2d,
)

pytestmark = pytest.mark.usefixtures("suppress_sibc_fit_plots")


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("invariant,normal", ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
@pytest.mark.parametrize("resistance", (np.inf, 5.0, "foster"))
def test_reduced_boundary_and_modal_build(tmp_path, polarization, invariant, normal, resistance):
    _, grid = run_2d(
        tmp_path / "grid",
        polarization=polarization,
        invariant=invariant,
        normal=normal,
        resistance=resistance,
        active=True,
        geometry_only=True,
        steps=100,
    )
    system = grid.impedance_surfaces
    assert system.edge_count > 0
    assert np.all(system.edge_info[:, invariant + 1] == (1 if polarization == "TE" else 0))
    assert not np.any(system.port_normal[:, 0] == invariant)
    electric = {invariant} if polarization == "TM" else set(range(3)) - {invariant}
    magnetic = set(range(3)) - {invariant} if polarization == "TM" else {invariant}
    assert set(system.edge_info[:, 0]) <= electric
    assert set(system.h_info[:, 0]) <= magnetic
    assert grid.dt > 0
    solver = grid.eigenmodesources[0].mode_solver
    assert solver.surface_boundary_rows
    free = solver.free_scalar_mask
    operator = solver.operator[free, :][:, free]
    vectors = solver.eigenvectors[free]
    residual = operator @ vectors - vectors * solver.eigenvalues
    assert (
        np.linalg.norm(residual) / (np.linalg.norm(operator.toarray()) * np.linalg.norm(vectors))
        < 1e-9
    )
    if resistance == np.inf:
        transverse_symbol = 0.0 if polarization == "TM" else 2 * np.sin(np.pi / 32) / 1e-3
        expected = np.sqrt(1 - (transverse_symbol / solver.operator_k0) ** 2)
        np.testing.assert_allclose(solver.operator_neff[0], expected, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("resistance", (np.inf, 5.0, "foster"))
def test_virtual_continuation(tmp_path, polarization, resistance):
    options = dict(polarization=polarization, resistance=resistance, steps=600)
    reference, _ = run_2d(tmp_path / "reference", **options)
    actual, grid = run_2d(tmp_path / "virtual", virtual=True, **options)
    scale = np.max(abs(reference), axis=-1, keepdims=True)
    scale = np.maximum(scale, np.max(scale) * 1e-12)
    assert np.max(abs(actual - reference) / scale) < 2e-8
    guide = grid.virtual_waveguides[0]
    assert guide.aux_grid.impedance_surfaces.edge_count > 0
    assert guide.aux_grid.impedance_surfaces.pml_edge_count > 0
    pml_histories = [
        getattr(slab, name)
        for slab in guide.aux_grid.pmls["slabs"]
        for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
    ]
    assert any(np.any(history) for history in pml_histories)
    if resistance == "foster":
        assert np.any(guide.aux_grid.impedance_surfaces.state_y)
        assert not np.shares_memory(
            grid.impedance_surfaces.state_y, guide.aux_grid.impedance_surfaces.state_y
        )
    guide.reset_run_state()
    for name in FIELDS:
        assert not np.any(getattr(guide.aux_grid, name))
    assert not np.any(guide.aux_grid.impedance_surfaces.state_y)
    assert not np.any(guide.aux_grid.impedance_surfaces.state_p)
    assert guide.aux_grid.impedance_surfaces._pending_modal_delta is None
    for slab in guide.aux_grid.pmls["slabs"]:
        for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2"):
            assert not np.any(getattr(slab, name))


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("invariant,normal", ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
@pytest.mark.parametrize("direction", ("+", "-"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("active", (False, True))
def test_rotated_virtual_fields(
    tmp_path, polarization, invariant, normal, direction, precision, active
):
    options = dict(
        polarization=polarization,
        invariant=invariant,
        normal=normal,
        direction=direction,
        precision=precision,
        resistance="foster",
        active=active,
        steps=600,
    )
    reference, _ = run_2d(tmp_path / "reference", **options)
    actual, grid = run_2d(tmp_path / "virtual", virtual=True, **options)
    scale = np.max(abs(reference), axis=-1, keepdims=True)
    scale = np.maximum(scale, np.max(scale) * 1e-12)
    assert np.max(abs(actual - reference) / scale) < (2e-4 if precision == "single" else 2e-8)
    inactive = (
        set(FIELDS)
        - set(grid.virtual_waveguides[0].reduced.active_electric)
        - set(grid.virtual_waveguides[0].reduced.active_magnetic)
    )
    for name in inactive:
        assert not np.any(getattr(grid, name))


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("invariant,normal", ((0, 1), (1, 2), (2, 0)))
@pytest.mark.parametrize("resistance", (np.inf, 5.0, "foster"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_2d_matches_independent_3d_extrusion(
    tmp_path, polarization, invariant, normal, resistance, precision
):
    result = compare_invariant_3d(
        tmp_path,
        polarization=polarization,
        invariant=invariant,
        normal=normal,
        resistance=resistance,
        precision=precision,
    )
    assert result["passed"], result


@pytest.mark.parametrize("polarization", ("TE", "TM"))
def test_synthetic_spacing_does_not_change_fields_or_modes(tmp_path, polarization):
    reference, first = run_2d(
        tmp_path / "first", polarization=polarization, resistance="foster", active=True
    )
    actual, second = run_2d(
        tmp_path / "second",
        polarization=polarization,
        resistance="foster",
        active=True,
        spacing=3e-3,
    )
    assert first.dt == second.dt
    np.testing.assert_allclose(
        second.eigenmodesources[0].complex_neff, first.eigenmodesources[0].complex_neff, rtol=1e-12
    )
    assert np.max(abs(reference - actual)) / np.max(abs(reference)) < 1e-11


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_active_exact_pmc_including_tem_zero_components(tmp_path, polarization, precision):
    kw = dict(
        polarization=polarization,
        invariant=1,
        normal=0,
        active=True,
        resistance=np.inf,
        precision=precision,
    )
    reference, _ = run_2d(tmp_path / "physical", **kw)
    actual, _ = run_2d(tmp_path / "virtual", virtual=True, **kw)
    assert normalized_field_error(reference, actual) < (2e-4 if precision == "single" else 2e-8)


@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("direction", ("+", "-"))
def test_plain_pec_2d_virtual_guide(tmp_path, polarization, active, direction):
    kw = dict(polarization=polarization, resistance="pec", active=active, direction=direction)
    reference, _ = run_2d(tmp_path / "physical", **kw)
    actual, grid = run_2d(tmp_path / "virtual", virtual=True, **kw)
    assert grid.impedance_surfaces is None
    assert normalized_field_error(reference, actual) < 2e-8


def test_nonuniform_invariant_storage_is_rejected(tmp_path, monkeypatch):
    from gprMax.grid.fdtd_grid import FDTDGrid

    original = FDTDGrid._build_impedance_surfaces

    def nonuniform(grid):
        grid.solid[10, 3, 0] = next(m.numID for m in grid.materials if m.ID == "free_space")
        original(grid)

    monkeypatch.setattr(FDTDGrid, "_build_impedance_surfaces", nonuniform)
    with pytest.raises(ValueError, match="uniform across the invariant-axis"):
        run_2d(tmp_path / "bad", polarization="TE", geometry_only=True, steps=1)
