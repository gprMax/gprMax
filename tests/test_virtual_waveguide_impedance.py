"""Physical split-guide checks for extruded surface-impedance walls."""

import copy

import numpy as np
import pytest

from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import evaluate_surface_ade
from testing.validation.impedance_surface.virtual_waveguide import (
    active_guide,
    compare_guides,
    run_guide,
)


@pytest.fixture(autouse=True)
def suppress_unused_fit_plots(monkeypatch):
    import gprMax.user_objects.cmds_multiuse as commands

    monkeypatch.setattr(commands, "plot_good_conductor_surface_impedance_fit", lambda **kwargs: None)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ("lossy", "debye", "lorentz", "drude"))
@pytest.mark.parametrize("normal_axis,direction,precision", (
    (0, "+", "double"),
    pytest.param(1, "-", "double", marks=pytest.mark.slow),
    (2, "+", "single"),
))
def test_mixed_bulk_sibc_virtual_guide_matches_physical_continuation(
    tmp_path, kind, normal_axis, direction, precision,
):
    options = dict(resistance="copper", host_kind=kind, layered=True,
                   normal_axis=normal_axis, direction=direction, precision=precision, steps=600,
                   formulation="MRIPML" if direction == "-" else "HORIPML")
    reference, _ = run_guide(tmp_path / "reference", virtual=False, **options)
    actual, grid = run_guide(tmp_path / "split", virtual=True, **options)
    weights = np.asarray((1, 1, 1, 376.730313668, 376.730313668, 376.730313668))[None, :, None]
    scale = np.max(np.abs(reference * weights))
    assert scale > 0
    assert np.max(np.abs((actual - reference) * weights)) / scale < (2e-8 if precision == "double" else 2e-4)
    guide = grid.virtual_waveguides[0]
    system = guide.aux_grid.impedance_surfaces
    assert system.pml_edge_count > 0
    assert np.any(system.state_y)
    if kind != "lossy":
        assert np.any(system.state_p)
        assert guide._dispersive_aperture
        assert any(np.any(getattr(guide.aux_grid, name)) for name in ("Tx", "Ty", "Tz"))
        assert not np.shares_memory(system.state_p, grid.impedance_surfaces.state_p)
    guide.reset_run_state()
    assert not np.any(system.state_y)
    assert not np.any(system.state_p)
    if guide.aux_grid.maxpoles:
        assert not any(np.any(getattr(guide.aux_grid, name)) for name in ("Tx", "Ty", "Tz"))


@pytest.mark.integration
@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("invariant,normal", ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
def test_dispersive_sibc_2d_virtual_guide_matches_physical_continuation(
    tmp_path, polarization, invariant, normal,
):
    from testing.validation.impedance_surface.validate_2d import normalized_field_error, run_2d

    options = dict(polarization=polarization, invariant=invariant, normal=normal,
                   resistance="foster", host_kind="debye" if normal % 2 else "lorentz",
                   direction="+" if invariant % 2 else "-", active=normal == 0, steps=600)
    reference, _ = run_2d(tmp_path / "reference", **options)
    actual, grid = run_2d(tmp_path / "split", virtual=True, **options)
    assert normalized_field_error(reference, actual) < 2e-8
    guide = grid.virtual_waveguides[0]
    assert guide._dispersive_aperture
    assert np.any(guide.aux_grid.impedance_surfaces.state_p)


@pytest.mark.parametrize("missing", ("aperture", "surface_bulk"))
def test_dispersive_virtual_guide_comparison_detects_missing_history(tmp_path, monkeypatch, missing):
    from gprMax.virtual_waveguide import VirtualWaveguide
    from testing.validation.impedance_surface.validate_2d import normalized_field_error, run_2d

    options = dict(polarization="TM", resistance="foster", host_kind="lorentz", steps=600)
    reference, _ = run_2d(tmp_path / "reference", **options)
    if missing == "aperture":
        monkeypatch.setattr(VirtualWaveguide, "_complete_dispersive_aperture", lambda *args: None)
    else:
        original = VirtualWaveguide._deposit_impedance_aperture_electric

        def drop_history(guide):
            original(guide)
            guide.aux_grid.impedance_surfaces.state_p.fill(0)

        monkeypatch.setattr(VirtualWaveguide, "_deposit_impedance_aperture_electric", drop_history)
    actual, _ = run_2d(tmp_path / "split", virtual=True, **options)
    assert normalized_field_error(reference, actual) > 1e-6


@pytest.mark.integration
@pytest.mark.parametrize("kind,direction", (("lossy", "+"), ("lorentz", "-")))
def test_active_bulk_sibc_virtual_guide_matches_physical_source(tmp_path, kind, direction):
    from testing.validation.impedance_surface.validate_2d import normalized_field_error

    options = dict(resistance="copper", host_kind=kind, layered=True,
                   direction=direction, active=True, steps=600)
    reference, _ = run_guide(tmp_path / "reference", virtual=False, **options)
    actual, _ = run_guide(tmp_path / "split", virtual=True, **options)
    assert np.max(np.abs(reference)) > 0
    assert normalized_field_error(reference, actual) < 2e-8


@pytest.mark.parametrize("polarization", ("TE", "TM"))
def test_dispersive_bulk_virtual_guide_without_surface_rows(tmp_path, polarization):
    from testing.validation.impedance_surface.validate_2d import normalized_field_error, run_2d

    options = dict(polarization=polarization, resistance="pec", host_kind="lorentz", steps=600)
    reference, _ = run_2d(tmp_path / "reference", **options)
    actual, grid = run_2d(tmp_path / "split", virtual=True, **options)
    assert normalized_field_error(reference, actual) < 2e-8
    assert not grid.virtual_waveguides[0]._impedance_edges


def test_exact_pmc_frequency_response_has_zero_admittance():
    response = evaluate_surface_ade(
        frequency_hz=22e9, dt=1e-12, F=np.empty((0, 0)), G=(), L=(), Z0=np.inf
    )
    assert response.admittance == 0j
    assert np.isposinf(response.impedance.real)


@pytest.mark.parametrize("resistance", [np.inf, 1000.0, "copper"])
@pytest.mark.integration
def test_extruded_impedance_virtual_guide_matches_monolithic(tmp_path, resistance):
    reference, _ = run_guide(tmp_path / "reference", virtual=False, resistance=resistance)
    actual, grid = run_guide(tmp_path / "split", virtual=True, resistance=resistance)
    scale = np.max(np.abs(reference), axis=-1, keepdims=True)
    scale = np.maximum(scale, np.max(scale) * 1e-12)
    assert np.max(np.abs(actual - reference) / scale) < 2e-8
    guide = grid.virtual_waveguides[0]
    assert guide.aux_grid.impedance_surfaces.edge_count > 0
    assert len(guide.aux_grid.impedance_surfaces.pml_edge_indices) > 0
    assert len(grid.impedance_surfaces.virtual_frozen_edges) > 0
    assert np.all(np.isfinite(guide.aux_grid.impedance_surfaces.state_y))


@pytest.mark.parametrize(
    "normal_axis,direction", [(0, "+"), (0, "-"), (1, "+"), (1, "-"), (2, "-")]
)
@pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
@pytest.mark.integration
def test_impedance_aperture_closure_is_rotation_and_direction_invariant(
    tmp_path, normal_axis, direction, formulation
):
    result = compare_guides(
        tmp_path,
        resistance=1000.0,
        normal_axis=normal_axis,
        direction=direction,
        formulation=formulation,
    )
    assert result["passed"], result


@pytest.mark.parametrize("resistance", [np.inf, 1000.0, "copper"])
@pytest.mark.integration
def test_active_surface_source_and_histories_terminate_with_low_reflection(tmp_path, resistance):
    result = active_guide(tmp_path / "active", resistance=resistance, steps=1200)
    assert result["passed"], result


@pytest.mark.integration
def test_virtual_surface_histories_reset_and_require_opaque_window_padding(tmp_path):
    _, grid = run_guide(tmp_path / "state", virtual=True, resistance="copper", steps=400)
    guide = grid.virtual_waveguides[0]
    system = guide.aux_grid.impedance_surfaces
    names = ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
    assert np.any(system.state_y)
    assert any(
        np.any(getattr(slab, name)) for slab in guide.aux_grid.pmls["slabs"] for name in names
    )
    guide.reset_run_state()
    assert not np.any(system.state_y)
    assert not any(
        np.any(getattr(slab, name)) for slab in guide.aux_grid.pmls["slabs"] for name in names
    )
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert not np.any(getattr(guide.aux_grid, name))

    cropped = copy.copy(guide)
    cropped.v0 = 4  # The lower SIBC wall now coincides with the modal-window edge.
    with pytest.raises(ValueError, match="strictly inside"):
        cropped._validate_impedance_cross_section()
