"""``PML.calculate_update_coeffs`` — the eight coefficient arrays.

This is where the CFS profiles become numbers the Cython kernels multiply by.
Two formulations produce eight arrays each, all closed-form in ``e0``, ``dt``
and the three profiles, so every assertion here is exact arithmetic rather
than a shape check.

Both formulations are reimplemented longhand at the top of this file. That
duplication is deliberate: a test that calls the function under test to
compute its own expectation proves only that the function is deterministic.

**Why the non-default CFS matters.** With the stock ``CFS()`` — ``alpha``
constant 0, ``kappa`` constant 1 — several formula terms vanish, and two pairs
of coefficients collapse into each other (``ERA == ERB`` under HORIPML;
``ERB == ERE == 1`` under MRIPML). A suite that only ever used the defaults
would pass with whole terms deleted from the source. The tests below check
both: the collapsed default case, because that is what most models actually
run, *and* a fully populated CFS where every term contributes.
"""

import logging

import numpy as np
import pytest

from gprMax.pml import PML

from .conftest import DT

# Every coefficient array, in the order the source allocates them.
COEFF_NAMES = ["ERA", "ERB", "ERE", "ERF", "HRA", "HRB", "HRE", "HRF"]


def horipml_reference(e0, dt, alpha, kappa, sigma):
    """HORIPML coefficients for one field polarity, written out longhand.

    Takes the three already-scaled profile arrays for either the electric or
    the magnetic samples and returns ``(A, B, E, F)``.
    """
    tmp = (2 * e0 * kappa) + dt * (alpha * kappa + sigma)
    A = (2 * e0 + dt * alpha) / tmp
    B = (2 * e0 * kappa) / tmp
    E = ((2 * e0 * kappa) - dt * (alpha * kappa + sigma)) / tmp
    F = (2 * sigma * dt) / (kappa * tmp)
    return A, B, E, F


def mripml_reference(e0, dt, alpha, kappa, sigma):
    """MRIPML coefficients for one field polarity, written out longhand."""
    tmp = 2 * e0 + dt * alpha
    A = kappa + (dt * sigma) / tmp
    B = (2 * e0) / tmp
    E = ((2 * e0) - dt * alpha) / tmp
    F = (2 * sigma * dt) / tmp
    return A, B, E, F


def profiles_for(cfs, thickness):
    """The six scaled profile arrays a slab of this thickness will see.

    Mirrors the three ``calculate_values`` calls the source makes, so the
    reference formulas above can be fed exactly what the code feeds its own.
    """
    Ealpha, Halpha = cfs.calculate_values(thickness, cfs.alpha)
    Ekappa, Hkappa = cfs.calculate_values(thickness, cfs.kappa)
    Esigma, Hsigma = cfs.calculate_values(thickness, cfs.sigma)
    return (Ealpha, Ekappa, Esigma), (Halpha, Hkappa, Hsigma)


@pytest.fixture
def e0():
    from gprMax import config

    return config.sim_config.em_consts["e0"]


@pytest.fixture
def rich_cfs(make_cfs):
    """A CFS with all three terms switched on and distinct.

    ``alpha`` ramps linearly, ``kappa`` quadratically from 1 to 4, ``sigma``
    quartically with an explicit ``max`` so no auto-calculation intervenes.
    Nothing here is zero or one, so no term can drop out of a formula and
    still produce the right answer.
    """
    return make_cfs(
        alpha={"scalingprofile": "linear", "min": 0.0, "max": 0.02},
        kappa={"scalingprofile": "quadratic", "min": 1.0, "max": 4.0},
        sigma={"scalingprofile": "quartic", "min": 0.0, "max": 6.0},
    )


class TestArrayShapesAndDtypes:
    @pytest.mark.parametrize("name", COEFF_NAMES)
    def test_shape_is_cfs_order_by_thickness(self, make_pml, name):
        """Expects ``(len(CFS), thickness)`` — one row per CFS term, one
        column per PML cell. (8 parameter sets)"""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert getattr(pml, name).shape == (1, 4)

    @pytest.mark.parametrize("name", COEFF_NAMES)
    def test_dtype_matches_the_configured_precision(self, make_pml, name):
        """Expects ``float64`` under the double-precision fixture, matching the
        fused type the Cython kernels are compiled for. (8 parameter sets)"""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert getattr(pml, name).dtype == np.float64

    @pytest.mark.parametrize("thickness", [1, 2, 5, 8])
    def test_column_count_follows_thickness(self, make_pml, thickness):
        """Expects one coefficient per cell of depth. (4 parameter sets)"""
        pml = make_pml(thickness=thickness)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA.shape == (1, thickness)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_row_count_follows_the_cfs_order(self, make_pml_grid, make_cfs, order):
        """Expects a row per CFS term, so a two-pole PML gets two rows.
        (3 parameter sets)"""
        cfs = [make_cfs(kappa={"min": 1.0}) for _ in range(order)]
        g = make_pml_grid(cfs=cfs)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA.shape == (order, 4)

    @pytest.mark.parametrize("name", COEFF_NAMES)
    def test_recalculating_replaces_the_arrays(self, make_pml, name):
        """Expects fresh allocations on each call rather than in-place
        updates. (8 parameter sets)"""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        first = getattr(pml, name)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert getattr(pml, name) is not first


class TestSigmamaxAutoCalculation:
    def test_unset_sigma_max_is_derived_from_the_material(self, make_pml):
        """Expects the ``None`` sentinel on a stock ``CFS`` to be replaced by
        the closed-form optimum for the backing material."""
        pml = make_pml(thickness=4)
        assert pml.CFS[0].sigma.max is None
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max == pytest.approx(
            0.8 * 5 / (self._z0() * pml.d)
        )

    @staticmethod
    def _z0():
        from gprMax import config

        return config.sim_config.em_consts["z0"]

    def test_an_explicit_sigma_max_is_left_alone(self, make_pml_grid, make_cfs):
        """Expects a user-supplied ``sigma.max`` to survive: the guard is
        ``if not cfs.sigma.max``, so any truthy value suppresses the
        auto-calculation."""
        g = make_pml_grid(cfs=[make_cfs(sigma={"max": 3.5})])
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max == 3.5

    def test_the_backing_material_changes_the_result(self, make_pml):
        """Expects a PML backing ``er = 4`` to derive half the ``sigma.max`` of
        one backing free space."""
        vacuum = make_pml(thickness=4)
        vacuum.calculate_update_coeffs(1.0, 1.0)
        soil = make_pml(thickness=4)
        soil.calculate_update_coeffs(4.0, 1.0)
        assert soil.CFS[0].sigma.max == pytest.approx(vacuum.CFS[0].sigma.max / 2)

    def test_calling_twice_does_not_recompute(self, make_pml):
        """Expects the second call to reuse the value cached on the CFS: after
        the first call ``sigma.max`` is truthy, so the guard no longer fires.

        This makes the method non-idempotent in an important way — passing a
        *different* material the second time silently has no effect on
        ``sigma.max``."""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        first = pml.CFS[0].sigma.max
        pml.calculate_update_coeffs(81.0, 1.0)
        assert pml.CFS[0].sigma.max == first

    def test_logs_the_derived_value_at_debug(self, make_pml, caplog):
        """Expects a debug record naming the slab and the value, once per CFS
        term."""
        pml = make_pml(pml_id="zmax", thickness=4)
        with caplog.at_level(logging.DEBUG, logger="gprMax.pml"):
            pml.calculate_update_coeffs(1.0, 1.0)
        assert "PML zmax: sigma.max set to" in caplog.text

    def test_the_derived_value_uses_the_slab_normal_spacing(self, make_pml):
        """Expects ``d`` — the spacing along the slab's own normal — in the
        denominator, so a y-slab on an anisotropic grid gets a different
        ``sigma.max`` from an x-slab."""
        from .conftest import DL_ANISO

        x_slab = make_pml(pml_id="x0", thickness=4, dl=DL_ANISO)
        y_slab = make_pml(pml_id="y0", thickness=4, dl=DL_ANISO)
        x_slab.calculate_update_coeffs(1.0, 1.0)
        y_slab.calculate_update_coeffs(1.0, 1.0)
        ratio = DL_ANISO[1] / DL_ANISO[0]
        assert y_slab.CFS[0].sigma.max == pytest.approx(
            x_slab.CFS[0].sigma.max / ratio
        )


class TestHoripmlAgainstTheClosedForm:
    """The default formulation, checked term by term."""

    @pytest.mark.parametrize("name", COEFF_NAMES)
    def test_every_coefficient_matches_the_longhand_formula(
        self, make_pml_grid, rich_cfs, e0, name
    ):
        """Expects agreement with an independently written HORIPML reference
        for a fully populated CFS — every one of the eight arrays.
        (8 parameter sets)"""
        g = make_pml_grid(cfs=[rich_cfs], formulation="HORIPML")
        pml = PML(g, "x0", "xminus", 0, 6, 0, 11, 0, 11)
        electric, magnetic = profiles_for(rich_cfs, 6)
        pml.calculate_update_coeffs(1.0, 1.0)
        expected = dict(
            zip(COEFF_NAMES[:4], horipml_reference(e0, DT, *electric))
        )
        expected.update(
            dict(zip(COEFF_NAMES[4:], horipml_reference(e0, DT, *magnetic)))
        )
        assert getattr(pml, name)[0] == pytest.approx(expected[name])

    def test_era_with_the_default_cfs_is_two_e0_over_tmp(self, make_pml, e0):
        """Expects ``ERA = 2·e0 / (2·e0 + dt·sigma)``: with ``alpha == 0`` the
        numerator's ``dt·alpha`` term vanishes and ``kappa == 1`` drops out of
        the denominator."""
        pml = make_pml(thickness=4)
        # ``sigma.max`` is the ``None`` sentinel until the call below derives
        # it, so the profile can only be reconstructed afterwards.
        pml.calculate_update_coeffs(1.0, 1.0)
        Esigma, _ = pml.CFS[0].calculate_values(4, pml.CFS[0].sigma)
        assert pml.ERA[0] == pytest.approx(2 * e0 / (2 * e0 + DT * Esigma))

    def test_era_equals_erb_for_the_default_cfs(self, make_pml):
        """Expects the two to coincide when ``alpha == 0`` and ``kappa == 1``.

        This is exactly why the formula tests above use ``rich_cfs``: with the
        defaults these two arrays are indistinguishable, so a suite built only
        on defaults could not tell the ``ERA`` and ``ERB`` expressions apart."""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA[0] == pytest.approx(pml.ERB[0])

    def test_era_and_erb_differ_once_alpha_is_on(self, make_pml_grid, rich_cfs):
        """Expects the collapse above to be a property of the defaults, not of
        the formulas."""
        g = make_pml_grid(cfs=[rich_cfs])
        pml = PML(g, "x0", "xminus", 0, 6, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA[0] != pytest.approx(pml.ERB[0])

    def test_ere_is_one_where_sigma_is_zero(self, make_pml):
        """Expects the innermost cell to be transparent: the quartic sigma
        profile starts at zero, so there ``ERE == 1`` and the PML applies no
        correction at all."""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERE[0, 0] == pytest.approx(1.0)

    def test_erf_is_zero_where_sigma_is_zero(self, make_pml):
        """Expects no loss term at the inner face, for the same reason."""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERF[0, 0] == pytest.approx(0.0)

    def test_erf_grows_outward(self, make_pml):
        """Expects the loss coefficient to increase monotonically with depth —
        the whole point of the graded ramp."""
        pml = make_pml(thickness=6)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert np.all(np.diff(pml.ERF[0]) > 0)

    def test_ere_shrinks_outward(self, make_pml):
        """Expects the retention coefficient to fall as absorption rises."""
        pml = make_pml(thickness=6)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert np.all(np.diff(pml.ERE[0]) < 0)

    def test_magnetic_coefficients_differ_from_electric(self, make_pml):
        """Expects the two sets to disagree, because the profiles they are
        built from are sampled half a cell apart."""
        pml = make_pml(thickness=4)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERF[0] != pytest.approx(pml.HRF[0])


class TestMripmlAgainstTheClosedForm:
    """The multipole formulation — a different algebra from the same profiles."""

    @pytest.mark.parametrize("name", COEFF_NAMES)
    def test_every_coefficient_matches_the_longhand_formula(
        self, make_pml_grid, rich_cfs, e0, name
    ):
        """Expects agreement with an independently written MRIPML reference for
        a fully populated CFS. (8 parameter sets)"""
        g = make_pml_grid(cfs=[rich_cfs], formulation="MRIPML")
        pml = PML(g, "x0", "xminus", 0, 6, 0, 11, 0, 11)
        electric, magnetic = profiles_for(rich_cfs, 6)
        pml.calculate_update_coeffs(1.0, 1.0)
        expected = dict(zip(COEFF_NAMES[:4], mripml_reference(e0, DT, *electric)))
        expected.update(
            dict(zip(COEFF_NAMES[4:], mripml_reference(e0, DT, *magnetic)))
        )
        assert getattr(pml, name)[0] == pytest.approx(expected[name])

    def test_erb_is_exactly_one_for_the_default_cfs(self, make_pml_grid):
        """Expects ``ERB = 2·e0 / (2·e0 + dt·alpha) == 1`` when ``alpha`` is
        zero everywhere."""
        g = make_pml_grid(formulation="MRIPML")
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERB[0] == pytest.approx(np.ones(4))

    def test_ere_is_exactly_one_for_the_default_cfs(self, make_pml_grid):
        """Expects ``ERE = (2·e0 - dt·alpha) / (2·e0 + dt·alpha) == 1`` for the
        same reason."""
        g = make_pml_grid(formulation="MRIPML")
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERE[0] == pytest.approx(np.ones(4))

    def test_era_starts_at_kappa(self, make_pml_grid):
        """Expects ``ERA = kappa + dt·sigma/(2·e0)``, which at the transparent
        inner face (``sigma == 0``) is exactly ``kappa``, i.e. 1 by default."""
        g = make_pml_grid(formulation="MRIPML")
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA[0, 0] == pytest.approx(1.0)

    def test_erf_is_sigma_dt_over_e0_for_the_default_cfs(self, make_pml_grid, e0):
        """Expects ``ERF = 2·sigma·dt / (2·e0) == sigma·dt/e0`` once the
        ``alpha`` term drops out of the denominator."""
        g = make_pml_grid(formulation="MRIPML")
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        cfs = pml.CFS[0]
        pml.calculate_update_coeffs(1.0, 1.0)
        Esigma, _ = cfs.calculate_values(4, cfs.sigma)
        assert pml.ERF[0] == pytest.approx(Esigma * DT / e0)

    def test_the_two_formulations_disagree(self, make_pml_grid, rich_cfs):
        """Expects genuinely different numbers from the same CFS — otherwise
        the formulation switch would be decorative."""
        results = {}
        for formulation in ("HORIPML", "MRIPML"):
            cfs = rich_cfs
            cfs.sigma.max = 6.0
            g = make_pml_grid(cfs=[cfs], formulation=formulation)
            pml = PML(g, "x0", "xminus", 0, 6, 0, 11, 0, 11)
            pml.calculate_update_coeffs(1.0, 1.0)
            results[formulation] = pml.ERA[0].copy()
        assert results["HORIPML"] != pytest.approx(results["MRIPML"])


class TestScalingWithDt:
    def test_a_zero_time_step_makes_the_pml_transparent(self, make_pml_grid):
        """Expects ``A == B == E == 1`` and ``F == 0`` when ``dt == 0``: every
        correction term carries a factor of ``dt``, so nothing is absorbed.

        A useful degenerate control — it isolates the ``dt``-free part of each
        formula."""
        g = make_pml_grid(dt=0.0)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA[0] == pytest.approx(np.ones(4))
        assert pml.ERB[0] == pytest.approx(np.ones(4))
        assert pml.ERE[0] == pytest.approx(np.ones(4))
        assert pml.ERF[0] == pytest.approx(np.zeros(4))

    def test_erf_is_linear_in_dt_for_small_steps(self, make_pml_grid, make_cfs):
        """Expects doubling ``dt`` to roughly double ``ERF`` while
        ``dt·sigma`` stays small against ``2·e0`` — the numerator is linear in
        ``dt`` and the denominator barely moves."""
        values = {}
        for dt in (1e-15, 2e-15):
            g = make_pml_grid(dt=dt, cfs=[make_cfs(sigma={"max": 1.0})])
            pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
            pml.calculate_update_coeffs(1.0, 1.0)
            values[dt] = pml.ERF[0, -1]
        assert values[2e-15] == pytest.approx(2 * values[1e-15], rel=1e-3)


class TestMultipole:
    def test_each_cfs_term_fills_its_own_row(self, make_pml_grid, make_cfs):
        """Expects two CFS terms with different sigma maxima to produce two
        distinct rows, in list order."""
        cfs = [
            make_cfs(kappa={"min": 0.5}, sigma={"max": 1.0}),
            make_cfs(kappa={"min": 0.5}, sigma={"max": 9.0}),
        ]
        g = make_pml_grid(cfs=cfs)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERF[0] != pytest.approx(pml.ERF[1])
        assert np.all(pml.ERF[1, 1:] > pml.ERF[0, 1:])

    def test_rows_are_independent(self, make_pml_grid, make_cfs):
        """Expects the second term's row to match a single-term PML built from
        the same CFS — no cross-talk between poles."""
        shared = {"kappa": {"min": 1.0}, "sigma": {"max": 9.0}}
        two = make_pml_grid(cfs=[make_cfs(sigma={"max": 1.0}), make_cfs(**shared)])
        one = make_pml_grid(cfs=[make_cfs(**shared)])
        a = PML(two, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        b = PML(one, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        a.calculate_update_coeffs(1.0, 1.0)
        b.calculate_update_coeffs(1.0, 1.0)
        assert a.ERF[1] == pytest.approx(b.ERF[0])

    def test_a_debug_record_is_emitted_per_term(self, make_pml_grid, make_cfs, caplog):
        """Expects one ``sigma.max set to`` record for each CFS term."""
        cfs = [make_cfs(kappa={"min": 0.5}), make_cfs(kappa={"min": 0.5})]
        g = make_pml_grid(cfs=cfs)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        with caplog.at_level(logging.DEBUG, logger="gprMax.pml"):
            pml.calculate_update_coeffs(1.0, 1.0)
        assert caplog.text.count("sigma.max set to") == 2
