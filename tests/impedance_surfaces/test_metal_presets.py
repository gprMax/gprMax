"""Common-metal and conductivity surface-impedance fitting tests."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.surface_impedance_presets import (
    DESAI_ALUMINIUM_DOI,
    DESAI_REFRACTORY_METALS_DOI,
    MATULA_1979_DOI,
    METAL_SURFACE_PRESETS,
    fit_conductivity_surface_impedance,
    fit_metal_surface_impedance,
    get_metal_surface_preset,
    good_conductor_surface_impedance,
)
from gprMax.user_objects.cmds_multiuse import SurfaceImpedance

REFERENCE_PRESETS = (
    ("aluminium", 2.650e-8, DESAI_ALUMINIUM_DOI),
    ("copper", 1.676e-8, MATULA_1979_DOI),
    ("gold", 2.192e-8, MATULA_1979_DOI),
    ("molybdenum", 5.34e-8, DESAI_REFRACTORY_METALS_DOI),
    ("palladium", 10.54e-8, MATULA_1979_DOI),
    ("silver", 1.586e-8, MATULA_1979_DOI),
    ("tungsten", 5.28e-8, DESAI_REFRACTORY_METALS_DOI),
    ("zinc", 5.964e-8, DESAI_REFRACTORY_METALS_DOI),
)


@pytest.mark.parametrize("name,resistivity,source", REFERENCE_PRESETS)
def test_reference_temperature_bulk_resistivities_are_explicit(name, resistivity, source):
    preset = METAL_SURFACE_PRESETS[name]
    assert preset.resistivity_ohm_m == resistivity
    assert preset.reference_temperature_k == 293.0
    assert preset.source == source
    assert preset.conductivity_s_per_m == pytest.approx(1 / resistivity)


@pytest.mark.parametrize(
    "alias,expected",
    (
        ("Cu", "copper"),
        ("AG", "silver"),
        ("au", "gold"),
        ("Pd", "palladium"),
        ("Al", "aluminium"),
        ("aluminum", "aluminium"),
        ("W", "tungsten"),
        ("Zn", "zinc"),
        ("Mo", "molybdenum"),
    ),
)
def test_element_symbol_and_spelling_aliases(alias, expected):
    assert get_metal_surface_preset(alias).key == expected


@pytest.mark.parametrize("name", tuple(METAL_SURFACE_PRESETS))
def test_auto_foster_fit_is_passive_stable_and_accurate(name):
    fit = fit_metal_surface_impedance(name, 1e8, 1e10)
    model = SurfaceImpedanceModel(
        ID=name,
        A=fit.A,
        B=fit.B,
        C=fit.C,
        D=fit.D,
        fit_fmin_hz=fit.fmin_hz,
        fit_fmax_hz=fit.fmax_hz,
        preset=name,
        provenance=fit.preset.source,
        conductivity_s_per_m=fit.conductivity_s_per_m,
        fit_requested_order=fit.requested_order,
        fit_pole_count=fit.selected_pole_count,
        fit_tolerance=fit.tolerance,
        fit_max_relative_error=fit.max_relative_error,
        fit_rms_relative_error=fit.rms_relative_error,
    )
    frequencies = np.geomspace(fit.fmin_hz, fit.fmax_hz, 2001)
    target = good_conductor_surface_impedance(frequencies, fit.preset.conductivity_s_per_m)
    relative_error = np.abs(model.impedance(frequencies) / target - 1)

    assert fit.requested_order == "auto"
    assert fit.meets_tolerance
    assert fit.selected_pole_count == fit.order
    assert fit.attempts[-1].pole_count == fit.selected_pole_count
    assert fit.attempts[-1].max_relative_error <= fit.tolerance
    assert [attempt.pole_count for attempt in fit.attempts] == list(
        range(1, fit.selected_pole_count + 1)
    )
    assert all(attempt.max_relative_error > fit.tolerance for attempt in fit.attempts[:-1])
    assert np.all(fit.B > 0)
    assert np.all(fit.C < 0)
    assert np.all(np.linalg.eigvals(fit.A).real < 0)
    assert fit.D > 0
    assert relative_error.max() <= fit.tolerance
    minimum_resistance = np.min(model.impedance(np.geomspace(1, 1e15, 3001)).real)
    roundoff_tolerance = 100 * np.finfo(float).eps * max(1.0, abs(fit.D))
    assert minimum_resistance >= -roundoff_tolerance
    assert model.discretise(1e-12).Z0 > 0


def test_surface_impedance_python_api_builds_preset_and_conductivity_fits():
    preset = SurfaceImpedance(
        id="silver_wall",
        preset="Ag",
        fit_frequency_range=(1e7, 2e10),
        fit_order=12,
        fit_tolerance=0.02,
    )
    conductor = SurfaceImpedance(
        id="alloy_wall",
        conductivity=3.2e7,
        fit_frequency_range=(1e7, 2e10),
        fit_order="auto",
    )

    np.testing.assert_allclose(preset.B, -preset.C)
    assert preset.preset == "silver"
    assert preset.fit_frequency_range == (1e7, 2e10)
    assert preset.fit_pole_count == 12
    assert preset.fit_max_relative_error < 0.02
    assert conductor.preset is None
    assert conductor.conductivity == 3.2e7
    assert conductor.fit_order == "auto"
    assert conductor.fit_meets_tolerance


def _surface_impedance_grid():
    return SimpleNamespace(
        name="",
        surface_impedance_models={},
        materials=[],
    )


def test_pure_resistance_build_warns_that_it_is_not_a_physical_material(caplog):
    surface = SurfaceImpedance(id="ideal_resistive_wall", resistance=50.0)

    with caplog.at_level(
        logging.WARNING,
        logger="gprMax.user_objects.cmds_multiuse",
    ):
        surface.build(_surface_impedance_grid())

    warnings = [
        record for record in caplog.records if "ideal_resistive_wall" in record.getMessage()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "frequency independent and purely real" in message
    assert "idealized boundary condition" in message
    assert "not a complete physical material model" in message
    assert "dispersive and generally have a reactive component" in message
    assert "fitted metal preset or conductivity model" in message


def test_fitted_metal_surface_does_not_emit_pure_resistance_warning(caplog):
    surface = SurfaceImpedance(
        id="copper_wall",
        preset="copper",
        fit_frequency_range=(8e9, 12e9),
    )

    with caplog.at_level(
        logging.WARNING,
        logger="gprMax.user_objects.cmds_multiuse",
    ):
        surface.build(_surface_impedance_grid())

    assert not any(
        "idealized boundary condition" in record.getMessage() for record in caplog.records
    )


def test_hash_api_accepts_new_explicit_source_forms():
    objects = get_user_objects(
        [
            "#surface_impedance: wall resistance 50\n",
            "#surface_impedance: cu_wall preset copper 8e9 12e9 auto 2e-3 y\n",
            "#surface_impedance: alloy conductivity 3.2e7 8e9 12e9 12\n",
        ],
        checkessential=False,
    )

    assert [type(item) for item in objects] == [SurfaceImpedance] * 3
    assert objects[0].resistance == 50
    assert objects[0].fit_result is None
    assert objects[1].preset == "copper"
    assert objects[1].fit_frequency_range == (8e9, 12e9)
    assert objects[1].fit_order == "auto"
    assert objects[1].fit_tolerance == 2e-3
    assert objects[1].plot_fit
    assert objects[2].preset is None
    assert objects[2].conductivity == 3.2e7
    assert objects[2].fit_pole_count == 12
    assert not objects[2].plot_fit


def test_surface_impedance_string_is_valid_new_hash_syntax():
    resistance = SurfaceImpedance(id="wall", resistance=50)
    copper = SurfaceImpedance(
        id="cu",
        preset="copper",
        fit_frequency_range=(8e9, 12e9),
        fit_order="auto",
        fit_tolerance=2e-3,
        plot_fit=True,
    )

    assert str(resistance) == "#surface_impedance: wall resistance 50"
    assert str(copper) == ("#surface_impedance: cu preset copper 8e+09 1.2e+10 auto 0.002 y")
    rebuilt = get_user_objects(
        [f"{resistance}\n", f"{copper}\n"],
        checkessential=False,
    )
    assert rebuilt[0].resistance == resistance.resistance
    assert rebuilt[1].preset == copper.preset
    assert rebuilt[1].plot_fit is copper.plot_fit


@pytest.mark.parametrize(
    "command",
    (
        "#surface_impedance: wall 50\n",
        "#surface_impedance: cu copper 8e9 12e9\n",
        "#surface_impedance: cu preset copper\n",
        "#surface_impedance: cu preset copper 8e9 12e9 auto 2e-3 plot\n",
    ),
)
def test_old_or_invalid_hash_forms_are_rejected(command):
    with pytest.raises((TypeError, ValueError)):
        get_user_objects([command], checkessential=False)


def test_fitted_sources_require_a_band_and_expert_coefficients_are_not_public():
    with pytest.raises(ValueError, match="fit_frequency_range"):
        SurfaceImpedance(id="cu", preset="copper")
    with pytest.raises(ValueError, match="fit_frequency_range"):
        SurfaceImpedance(id="alloy", conductivity=3.2e7)
    with pytest.raises(TypeError, match="unexpected keyword argument 'D'"):
        SurfaceImpedance(id="custom", D=50)


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"resistance": 50, "preset": "copper"},
        {"preset": "copper", "conductivity": 3.2e7},
    ),
)
def test_surface_impedance_requires_exactly_one_source(options):
    with pytest.raises(ValueError, match="exactly one"):
        SurfaceImpedance(id="invalid", **options)


@pytest.mark.parametrize(
    "fit_frequency_range",
    (
        (0, 12e9),
        (12e9, 8e9),
        (8e9, np.inf),
        (np.nan, 12e9),
        (8e9, 10e9, 12e9),
    ),
)
def test_surface_impedance_rejects_invalid_fit_bands(fit_frequency_range):
    with pytest.raises(ValueError, match="fit|band|frequencies"):
        SurfaceImpedance(
            id="invalid",
            preset="copper",
            fit_frequency_range=fit_frequency_range,
        )


@pytest.mark.parametrize("fit_order", (True, 0, 65, "twelve"))
def test_surface_impedance_rejects_invalid_fit_orders(fit_order):
    with pytest.raises(ValueError, match="fit order"):
        SurfaceImpedance(
            id="invalid",
            preset="copper",
            fit_frequency_range=(8e9, 12e9),
            fit_order=fit_order,
        )


@pytest.mark.parametrize("fit_tolerance", (0, -1, np.inf, np.nan))
def test_surface_impedance_rejects_invalid_fit_tolerances(fit_tolerance):
    with pytest.raises(ValueError, match="fit tolerance"):
        SurfaceImpedance(
            id="invalid",
            preset="copper",
            fit_frequency_range=(8e9, 12e9),
            fit_tolerance=fit_tolerance,
        )


def test_unknown_metal_preset_is_rejected_with_choices():
    with pytest.raises(ValueError, match="aluminium.*copper.*gold.*molybdenum"):
        SurfaceImpedance(
            id="bad",
            preset="unobtainium",
            fit_frequency_range=(1e8, 1e10),
        )


def test_cached_preset_arrays_cannot_be_mutated_or_made_writable():
    fit = fit_metal_surface_impedance("copper", 1e8, 1e10)
    original = float(fit.A[0, 0])
    with pytest.raises(ValueError, match="read-only"):
        fit.A[0, 0] = original * 2
    with pytest.raises(ValueError):
        fit.A.setflags(write=True)
    with pytest.raises(ValueError):
        fit.A.base.setflags(write=True)
    assert fit_metal_surface_impedance("copper", 1e8, 1e10).A[0, 0] == original


def test_explicit_high_order_conductivity_fit_is_supported_and_reported():
    fit = fit_conductivity_surface_impedance(5.8e7, 1e6, 1e11, 64, 2e-3)
    assert fit.requested_order == 64
    assert len(fit.attempts) == 1
    assert fit.selected_pole_count == fit.order == 64
    assert np.isfinite(fit.max_relative_error)


def test_auto_order_tracks_runtime_poles_and_bandwidth():
    ultra_narrow = fit_metal_surface_impedance("copper", 10e9, 10.1e9)
    quarter_octave = fit_metal_surface_impedance("copper", 120e9, 150e9)
    narrow = fit_metal_surface_impedance("copper", 8e9, 12e9)
    octave = fit_metal_surface_impedance("copper", 8e9, 16e9)
    two_decades = fit_metal_surface_impedance("copper", 1e8, 1e10)

    assert ultra_narrow.selected_pole_count == 1
    assert quarter_octave.selected_pole_count == 2
    assert narrow.selected_pole_count == 2
    assert octave.selected_pole_count == 3
    assert two_decades.selected_pole_count == 8
    assert ultra_narrow.order < narrow.order < octave.order < two_decades.order
    assert all(
        fit.meets_tolerance for fit in (ultra_narrow, quarter_octave, narrow, octave, two_decades)
    )
    assert narrow.max_relative_error < 1.6e-3
    assert octave.attempts[1].max_relative_error > octave.tolerance
    assert octave.max_relative_error < 5e-4


def test_tighter_auto_tolerance_never_selects_fewer_runtime_poles():
    loose = fit_metal_surface_impedance("copper", 8e9, 12e9, "auto", 2e-3)
    tight = fit_metal_surface_impedance("copper", 8e9, 12e9, "auto", 5e-4)

    assert loose.selected_pole_count == 2
    assert tight.selected_pole_count == 3
    assert tight.selected_pole_count >= loose.selected_pole_count
    assert loose.meets_tolerance and tight.meets_tolerance
    assert loose.attempts == tight.attempts[: len(loose.attempts)]


def test_explicit_order_fit_is_independent_of_requested_tolerance():
    loose = fit_conductivity_surface_impedance(
        3.2e7,
        120e9,
        150e9,
        1,
        0.1,
    )
    strict = fit_conductivity_surface_impedance(
        3.2e7,
        120e9,
        150e9,
        1,
        1e-6,
    )

    assert loose.selected_pole_count == strict.selected_pole_count == 1
    assert 0.02 < loose.max_relative_error < 0.03
    assert loose.max_relative_error == strict.max_relative_error
    assert loose.rms_relative_error == strict.rms_relative_error
    assert loose.attempts == strict.attempts
    assert loose.D == strict.D
    for attribute in ("A", "B", "C"):
        np.testing.assert_array_equal(
            getattr(loose, attribute),
            getattr(strict, attribute),
        )


def test_normalised_auto_fit_is_scale_and_conductivity_invariant():
    low = fit_conductivity_surface_impedance(1.0e7, 8e9, 12e9)
    high = fit_conductivity_surface_impedance(8.0e7, 80e9, 120e9)

    assert low.selected_pole_count == high.selected_pole_count == 2
    assert low.max_relative_error == high.max_relative_error
    assert low.rms_relative_error == high.rms_relative_error
    assert low.attempts == high.attempts
    np.testing.assert_allclose(
        -np.diag(low.A) / (2 * np.pi * low.fmin_hz),
        -np.diag(high.A) / (2 * np.pi * high.fmin_hz),
        rtol=4 * np.finfo(np.float64).eps,
        atol=0,
    )

    frequency_ratio = np.geomspace(1, 1.5, 101)
    low_response = SurfaceImpedanceModel("low", A=low.A, B=low.B, C=low.C, D=low.D).impedance(
        low.fmin_hz * frequency_ratio
    )
    high_response = SurfaceImpedanceModel("high", A=high.A, B=high.B, C=high.C, D=high.D).impedance(
        high.fmin_hz * frequency_ratio
    )
    low_scale = good_conductor_surface_impedance((low.fmin_hz,), 1.0e7)[0]
    high_scale = good_conductor_surface_impedance((high.fmin_hz,), 8.0e7)[0]
    np.testing.assert_allclose(low_response / low_scale, high_response / high_scale)


def test_normalised_fit_is_deterministic_across_equivalent_public_requests():
    by_name = fit_metal_surface_impedance("copper", 8e9, 12e9)
    by_alias = fit_metal_surface_impedance("Cu", 8e9, 12e9)
    by_conductivity = fit_conductivity_surface_impedance(
        by_name.conductivity_s_per_m,
        8e9,
        12e9,
    )

    assert by_name.attempts == by_alias.attempts == by_conductivity.attempts
    assert by_name.D == by_alias.D == by_conductivity.D
    for attribute in ("A", "B", "C"):
        np.testing.assert_array_equal(
            getattr(by_name, attribute),
            getattr(by_alias, attribute),
        )
        np.testing.assert_array_equal(
            getattr(by_name, attribute),
            getattr(by_conductivity, attribute),
        )


def test_fit_rejects_non_microwave_or_non_good_conductor_targets():
    with pytest.raises(ValueError, match="microwave band"):
        fit_metal_surface_impedance("copper", 100e12, 200e12)
    with pytest.raises(ValueError, match="good-conductor requirement"):
        fit_conductivity_surface_impedance(1.0, 100e9, 200e9)


def test_foster_pruning_is_scale_invariant_for_small_impedance_coefficients():
    fit = fit_conductivity_surface_impedance(1e30, 8e9, 12e9, 12, 2e-3)
    frequencies = np.geomspace(fit.fmin_hz, fit.fmax_hz, 2001)
    target = good_conductor_surface_impedance(
        frequencies,
        fit.conductivity_s_per_m,
    )
    model = SurfaceImpedanceModel(
        "tiny_impedance",
        A=fit.A,
        B=fit.B,
        C=fit.C,
        D=fit.D,
        fit_fmin_hz=fit.fmin_hz,
        fit_fmax_hz=fit.fmax_hz,
    )

    assert np.max(np.abs(model.impedance(frequencies) / target - 1)) <= 2e-3


def test_surface_model_detaches_and_freezes_caller_owned_arrays():
    A = np.asarray(((-2.0,),))
    B = np.asarray((3.0,))
    C = np.asarray((4.0,))
    model = SurfaceImpedanceModel("detached", A=A, B=B, C=C, D=5.0)
    A[0, 0] = -7.0
    B[0] = 8.0
    C[0] = 9.0
    np.testing.assert_array_equal(model.A, ((-2.0,),))
    np.testing.assert_array_equal(model.B, (3.0,))
    np.testing.assert_array_equal(model.C, (4.0,))
    with pytest.raises(ValueError, match="read-only"):
        model.B[0] = 10.0
    with pytest.raises(ValueError):
        model.B.setflags(write=True)
    with pytest.raises(ValueError):
        model.B.base.setflags(write=True)


def test_dispersive_fit_band_is_enforced_but_constant_resistance_is_global():
    dynamic = SurfaceImpedanceModel(
        "dynamic",
        A=((-1.0,),),
        B=(1.0,),
        C=(1.0,),
        D=1.0,
        fit_fmin_hz=10.0,
        fit_fmax_hz=20.0,
    )
    dynamic.require_fit_frequency(10.0, purpose="test")
    dynamic.require_fit_frequency(20.0, purpose="test")
    with pytest.raises(ValueError, match="expand the fit band"):
        dynamic.require_fit_frequency(21.0, purpose="test")
    with pytest.raises(ValueError, match="bilinear-warped"):
        dynamic.require_fit_frequency(
            21.0,
            purpose="test",
            frequency_kind="bilinear-warped",
        )

    constant = SurfaceImpedanceModel(
        "constant",
        D=5.0,
        fit_fmin_hz=10.0,
        fit_fmax_hz=20.0,
    )
    constant.require_fit_frequency(1e12, purpose="test")
