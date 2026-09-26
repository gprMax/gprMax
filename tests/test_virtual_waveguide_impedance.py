"""Physical split-guide checks for extruded surface-impedance walls."""

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
def test_virtual_surface_histories_reset(tmp_path):
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


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis,direction", [(axis, sign) for axis in range(3) for sign in ("+", "-")])
@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("walls_on_rim", ("lower", "upper", "both"))
def test_sibc_wall_on_virtual_rim_matches_physical_pec_continuation(
    tmp_path, monkeypatch, normal_axis, direction, active, walls_on_rim,
):
    """A wall on the rim becomes PEC in the modal solve and time-domain guide."""
    import gprMax
    from testing.validation.impedance_surface import virtual_waveguide as validation
    from testing.validation.impedance_surface.validate_2d import normalized_field_error

    original_scene = validation.guide_scene
    transverse = tuple(axis for axis in range(3) if axis != normal_axis)
    basis = (*transverse, normal_axis)
    v0 = 4 if walls_on_rim in ("lower", "both") else 2
    v1 = 12 if walls_on_rim in ("upper", "both") else 14

    def point(local):
        result = np.zeros(3)
        result[list(basis)] = np.asarray(local) * 1e-3
        return tuple(result)

    def scene(**kwargs):
        result = original_scene(**kwargs)
        for obj in result.grid_objects:
            if isinstance(obj, gprMax.EigenmodePort):
                plane = obj.kwargs["p1"][normal_axis] / 1e-3
                obj.kwargs.update(p1=point((2, v0, plane)), p2=point((14, v1, plane)))
        if not kwargs["virtual"]:
            # Replace only the rear continuation's rim with real PEC plates;
            # the main-domain side retains its original lossy copper walls.
            low, high = (0, 24) if direction == "+" else (48, 72)
            for v in (v0, v1):
                result.add(gprMax.Plate(
                    p1=point((2, v, low)), p2=point((14, v, high)), material_id="pec"
                ))
        return result

    monkeypatch.setattr(validation, "guide_scene", scene)
    options = dict(resistance="copper", host_kind="lossy", steps=400,
                   normal_axis=normal_axis, direction=direction, active=active,
                   formulation="HORIPML" if direction == "+" else "MRIPML")
    reference, reference_grid = run_guide(tmp_path / "physical", virtual=False, **options)
    actual, grid = run_guide(tmp_path / "virtual", virtual=True, **options)
    assert np.max(np.abs(reference)) > 0
    assert normalized_field_error(reference, actual) < 2e-8

    guide = grid.virtual_waveguides[0]
    assert len(guide._impedance_window_edges) > len(guide._impedance_edges)
    assert bool(guide._impedance_edges) == (walls_on_rim != "both")
    for component in (transverse[0], normal_axis):
        field = getattr(guide.aux_grid, "E" + "xyz"[component])
        for v in (0, guide.nv):
            assert not np.any(np.take(field, v, axis=transverse[1]))
    # An active drive also exercises the modal fields copied to the auxiliary
    # source, rather than only passive time-domain aperture coupling.
    if active:
        assert guide.port.mode_solver.modal_complex_neff == pytest.approx(
            reference_grid.eigenmodesources[0].mode_solver.modal_complex_neff,
            rel=1e-10,
        )


@pytest.mark.integration
@pytest.mark.parametrize("normal_axis,direction", [(axis, sign) for axis in range(3) for sign in ("+", "-")])
@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("geometry", ("guide", "microstrip"))
def test_cropped_sibc_guide_matches_physical_pec_continuation(
    tmp_path, monkeypatch, normal_axis, direction, active, geometry,
):
    """A real PEC rim must give the same fields as the artificial cropped rim."""
    import gprMax
    from testing.validation.impedance_surface import virtual_waveguide as validation

    original_scene = validation.guide_scene
    transverse = tuple(axis for axis in range(3) if axis != normal_axis)
    basis = (*transverse, normal_axis)

    def point(local):
        result = np.zeros(3)
        result[list(basis)] = np.asarray(local) * 1e-3
        return tuple(result)

    def cropped_scene(**kwargs):
        scene = original_scene(**kwargs)
        if geometry == "microstrip":
            # Finite copper ground wider than the aperture, a lossy FR-4
            # substrate, and a narrower copper strip above it.
            scene.geometry_objects.clear()
            for lower, upper, material in (
                ((1, 3, 0), (15, 4, 72), "wall"),
                ((1, 4, 0), (15, 6, 72), "host"),
                ((7, 6, 0), (9, 7, 72), "wall"),
            ):
                scene.add(gprMax.Box(p1=point(lower), p2=point(upper), material_id=material))
            for obj in scene.grid_objects:
                if isinstance(obj, gprMax.Material) and obj.kwargs["id"] == "host":
                    obj.kwargs.update(er=4.4, se=0.005)
                if isinstance(obj, gprMax.HertzianDipole):
                    obj.kwargs["polarisation"] = "xyz"[transverse[1]]
                if isinstance(obj, gprMax.Waveform):
                    obj.kwargs["freq"] = 10e9
                if isinstance(obj, gprMax.EigenmodeBand):
                    obj.kwargs.update(fmin=10e9, fmax=10e9)
                if isinstance(obj, gprMax.EigenmodePort):
                    obj.kwargs["anchors"] = (10e9,)
        plane = 24 if direction == "+" else 48
        for obj in scene.grid_objects:
            if isinstance(obj, gprMax.EigenmodePort):
                port_plane = obj.kwargs["p1"][normal_axis] / 1e-3
                obj.kwargs.update(p1=point((5, 2, port_plane)), p2=point((11, 14, port_plane)))
        if not kwargs["virtual"]:
            # Explicit plates isolate precisely the same physical rear volume
            # as the auxiliary grid. Copper crosses both side plates.
            low, high = (0, plane) if direction == "+" else (plane, 72)
            for u in (5, 11):
                scene.add(gprMax.Plate(p1=point((u, 2, low)), p2=point((u, 14, high)), material_id="pec"))
            for v in (2, 14):
                scene.add(gprMax.Plate(p1=point((5, v, low)), p2=point((11, v, high)), material_id="pec"))
        return scene

    monkeypatch.setattr(validation, "guide_scene", cropped_scene)
    options = dict(resistance="copper", host_kind="lossy", layered=True, steps=400,
                   normal_axis=normal_axis, direction=direction, active=active)
    reference, _ = run_guide(tmp_path / "physical", virtual=False, **options)
    actual, grid = run_guide(tmp_path / "cropped", virtual=True, **options)
    weights = np.asarray((1, 1, 1, 376.730313668, 376.730313668, 376.730313668))[None, :, None]
    scale = np.max(np.abs(reference * weights))
    assert scale > 0
    assert np.max(np.abs((actual - reference) * weights)) / scale < 2e-8

    guide = grid.virtual_waveguides[0]
    assert len(guide._impedance_window_edges) > len(guide._impedance_edges) > 0
    # E normal to a cut face remains active; E tangent to it is clamped.
    rows = guide.aux_grid.impedance_surfaces.edge_info
    assert np.any(rows[:, 0] == transverse[0])
    for component in (transverse[1], normal_axis):
        for u in (0, guide.nu):
            assert not np.any((rows[:, 0] == component) & (rows[:, 1 + transverse[0]] == u))
            assert not np.any(np.take(getattr(guide.aux_grid, "E" + "xyz"[component]), u, axis=transverse[0]))
    assert np.any(guide.aux_grid.impedance_surfaces.state_y)
    frozen = grid.impedance_surfaces.virtual_frozen_edges
    assert not np.any((frozen[:, 0] == transverse[0]) & (frozen[:, 1 + transverse[0]] == guide.u1))

    # Neither extractor may accept an absent sample on a retained row.
    if geometry == "microstrip" and active and normal_axis == 0 and direction == "+":
        system = grid.impedance_surfaces
        edge = system.edge_info[guide._impedance_edges[0]]
        candidates = range(int(edge[4]), int(edge[4] + edge[5]))
        sample = next(i for i in candidates if edge[0] == normal_axis or system.h_info[i, 0] == normal_axis)
        with monkeypatch.context() as patch:
            corrupted = system.h_info.copy()
            corrupted[sample, 1 + transverse[0]] = guide.u0 - 1
            patch.setattr(system, "h_info", corrupted)
            with pytest.raises(ValueError, match="required magnetic DOF"):
                guide.port._build_surface_impedance_fdfd_boundary(grid)
            with pytest.raises(ValueError, match="required magnetic sample"):
                guide._build_impedance_auxiliary_system(guide.aux_grid)
