"""Common-metal surface-impedance preset tests."""

import numpy as np
import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.surface_impedance_presets import (
    MATULA_1979_DOI,
    METAL_SURFACE_PRESETS,
    fit_metal_surface_impedance,
    get_metal_surface_preset,
    good_conductor_surface_impedance,
)
from gprMax.user_objects.cmds_multiuse import SurfaceImpedance


@pytest.mark.parametrize(
    "name,resistivity",
    (("copper", 1.676e-8), ("gold", 2.192e-8), ("silver", 1.586e-8)),
)
def test_reference_temperature_bulk_resistivities_are_explicit(name, resistivity):
    preset = METAL_SURFACE_PRESETS[name]
    assert preset.resistivity_ohm_m == resistivity
    assert preset.reference_temperature_k == 293.0
    assert preset.source == MATULA_1979_DOI
    assert preset.conductivity_s_per_m == pytest.approx(1 / resistivity)


@pytest.mark.parametrize("alias,expected", (("Cu", "copper"), ("AG", "silver"), ("au", "gold")))
def test_element_symbol_aliases(alias, expected):
    assert get_metal_surface_preset(alias).key == expected


@pytest.mark.parametrize("name", tuple(METAL_SURFACE_PRESETS))
def test_default_foster_fit_is_passive_stable_and_accurate(name):
    fit = fit_metal_surface_impedance(name)
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
        fit_max_relative_error=fit.max_relative_error,
    )
    frequencies = np.geomspace(fit.fmin_hz, fit.fmax_hz, 2001)
    target = good_conductor_surface_impedance(
        frequencies, fit.preset.conductivity_s_per_m
    )
    relative_error = np.abs(model.impedance(frequencies) / target - 1)

    assert 0 < fit.order <= fit.candidate_order
    assert np.all(np.linalg.eigvals(fit.A).real < 0)
    assert fit.D > 0
    assert relative_error.max() < 0.002
    assert np.min(model.impedance(np.geomspace(1, 1e15, 3001)).real) >= 0
    assert model.discretise(1e-12).Z0 > 0


def test_surface_impedance_python_api_builds_balanced_preset_realization():
    surface = SurfaceImpedance(
        id="silver_wall",
        preset="Ag",
        fit_fmin_hz=1e7,
        fit_fmax_hz=2e10,
        fit_order=12,
    )
    np.testing.assert_allclose(surface.B, -surface.C)
    assert surface.preset == "silver"
    assert surface.fit_fmin_hz == 1e7
    assert surface.fit_fmax_hz == 2e10
    assert surface.fit_max_relative_error < 0.02


def test_hash_api_accepts_default_and_explicit_metal_fit_bands():
    objects = get_user_objects(
        [
            "#surface_impedance: cu_wall copper\n",
            "#surface_impedance: au_wall Au 1e7 2e10 12\n",
        ],
        checkessential=False,
    )
    assert [type(item) for item in objects] == [SurfaceImpedance, SurfaceImpedance]
    assert objects[0].preset == "copper"
    assert objects[1].preset == "gold"
    assert objects[1].fit_fmin_hz == 1e7
    assert objects[1].fit_fmax_hz == 2e10
    assert objects[1].fit_order == 12


def test_unknown_metal_preset_is_rejected_with_choices():
    with pytest.raises(ValueError, match="copper, gold, silver"):
        SurfaceImpedance(id="bad", preset="unobtainium")


def test_cached_preset_arrays_cannot_be_mutated_or_made_writable():
    fit = fit_metal_surface_impedance("copper")
    original = float(fit.A[0, 0])
    with pytest.raises(ValueError, match="read-only"):
        fit.A[0, 0] = original * 2
    with pytest.raises(ValueError):
        fit.A.setflags(write=True)
    with pytest.raises(ValueError):
        fit.A.base.setflags(write=True)
    assert fit_metal_surface_impedance("copper").A[0, 0] == original


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
