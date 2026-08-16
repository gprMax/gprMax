"""``CFSParameter`` and ``CFS`` — the PML absorption gradient.

The PML works by ramping absorption in gradually: a hard step from vacuum to
"very absorbing" is itself an impedance discontinuity, and an impedance
discontinuity is a mirror. Everything in this file is the arithmetic of that
ramp, and all of it is closed-form.

Two facts govern the expected values throughout, and both are worth stating
once here rather than repeating in thirty docstrings.

**The profile is sampled twice per cell.** ``calculate_values`` allocates
``thickness + 1`` samples, hands them to ``scaling_polynomial`` which builds a
single ``linspace`` of ``2n`` points, splits the even entries to the electric
profile and the odd entries to the magnetic one, then drops the final sample.
For ``thickness == 4`` and a linear profile that leaves E at ``0, ¼, ½, ¾``
and H at ``⅛, ⅜, ⅝, ⅞`` — H trailing E by half a cell, which is exactly the
Yee stagger.

**Reversal is not symmetric.** For a slab on the far side of the domain both
arrays are reversed, and the magnetic one is then rolled one element left.
That roll is the half-cell offset reasserting itself: reverse an E sample and
it lands where an E sample belongs, reverse an H sample and it lands half a
cell out.
"""

import numpy as np
import pytest

from gprMax.pml import CFS, CFSParameter

from .conftest import DL

# The nine polynomial profile names, paired with the order each denotes.
PROFILES = [
    ("constant", 0),
    ("linear", 1),
    ("quadratic", 2),
    ("cubic", 3),
    ("quartic", 4),
    ("quintic", 5),
    ("sextic", 6),
    ("septic", 7),
    ("octic", 8),
]


def polynomial_profile(thickness, order):
    """Reference implementation of ``scaling_polynomial``, written out longhand.

    Reproducing the formula independently is the point: an assertion that
    calls the function under test to compute its own expectation proves
    nothing.
    """
    n = thickness + 1
    tmp = (np.linspace(0, (n - 1) + 0.5, num=2 * n) / (n - 1)) ** order
    return tmp[0:-1:2], tmp[1::2]


class TestCFSParameterDefaults:
    def test_all_arguments_default(self):
        """Expects a bare ``CFSParameter`` to be an inert, unnamed, zero-valued
        polynomial parameter with no profile chosen."""
        p = CFSParameter()
        assert p.ID is None
        assert p.scaling == "polynomial"
        assert p.scalingprofile is None
        assert p.scalingdirection == "forward"
        assert p.min == 0
        assert p.max == 0

    @pytest.mark.parametrize(
        "name,value",
        [
            ("ID", "sigma"),
            ("scaling", "polynomial"),
            ("scalingprofile", "cubic"),
            ("scalingdirection", "reverse"),
            ("min", 0.5),
            ("max", 12.0),
        ],
    )
    def test_each_argument_is_stored_verbatim(self, name, value):
        """Expects every constructor argument to land on the attribute of the
        same name, unmodified. (6 parameter sets)"""
        assert getattr(CFSParameter(**{name: value}), name) == value


class TestScalingProfileTable:
    def test_there_are_exactly_nine_profiles(self):
        """Expects ``scalingprofiles`` to hold nine entries — a change here
        changes what every profile name means."""
        assert len(CFSParameter.scalingprofiles) == 9

    @pytest.mark.parametrize("name,order", PROFILES)
    def test_name_maps_to_its_polynomial_order(self, name, order):
        """Expects ``"linear" -> 1``, ``"quartic" -> 4``, and so on: the order
        is the position of the name in the sequence, starting at
        ``"constant" -> 0``. (9 parameter sets)"""
        assert CFSParameter.scalingprofiles[name] == order

    def test_scaling_directions_are_forward_and_reverse(self):
        """Expects exactly two directions, in that order."""
        assert CFSParameter.scalingdirections == ["forward", "reverse"]


class TestCFSDefaults:
    """The stock CFS: sigma does the absorbing, kappa and alpha are off."""

    def test_alpha_is_a_constant_zero(self):
        """Expects ``alpha`` constant with ``min == max == 0`` — the
        frequency-shift term is switched off by default."""
        cfs = CFS()
        assert cfs.alpha.ID == "alpha"
        assert cfs.alpha.scalingprofile == "constant"
        assert (cfs.alpha.min, cfs.alpha.max) == (0, 0)

    def test_kappa_is_a_constant_one(self):
        """Expects ``kappa`` constant with ``min == max == 1`` — a stretch
        factor of one is no stretching, so it too is off."""
        cfs = CFS()
        assert cfs.kappa.ID == "kappa"
        assert cfs.kappa.scalingprofile == "constant"
        assert (cfs.kappa.min, cfs.kappa.max) == (1, 1)

    def test_sigma_is_a_quartic_ramp_with_max_unset(self):
        """Expects ``sigma`` quartic, ``min == 0`` and ``max is None`` — the
        ``None`` is the sentinel that makes ``calculate_update_coeffs`` derive
        the optimum from the underlying material."""
        cfs = CFS()
        assert cfs.sigma.ID == "sigma"
        assert cfs.sigma.scalingprofile == "quartic"
        assert cfs.sigma.min == 0
        assert cfs.sigma.max is None

    def test_two_instances_do_not_share_parameters(self):
        """Expects each ``CFS`` to own its three parameters — a shared
        ``CFSParameter`` would let one PML slab's auto-computed ``sigma.max``
        leak into another's."""
        a, b = CFS(), CFS()
        assert a.sigma is not b.sigma
        a.sigma.max = 99.0
        assert b.sigma.max is None


class TestCalculateSigmamax:
    """``sigma_max = 0.8·(m+1) / (z0·d·sqrt(er·mr))``."""

    def test_matches_the_closed_form_for_the_default_quartic(self, make_cfs):
        """Expects the published optimum for a quartic profile in free space:
        ``0.8·5 / (z0·d)``."""
        from gprMax import config

        cfs = make_cfs()
        cfs.calculate_sigmamax(DL, 1.0, 1.0)
        z0 = config.sim_config.em_consts["z0"]
        assert cfs.sigma.max == pytest.approx(0.8 * 5 / (z0 * DL))

    @pytest.mark.parametrize("name,order", PROFILES)
    def test_numerator_follows_the_profile_order(self, make_cfs, name, order):
        """Expects the ``(m + 1)`` factor to come from ``sigma``'s own profile
        name, so the name-to-order table is pinned end to end. (9 parameter
        sets)"""
        from gprMax import config

        cfs = make_cfs(sigma={"scalingprofile": name})
        cfs.calculate_sigmamax(DL, 1.0, 1.0)
        z0 = config.sim_config.em_consts["z0"]
        assert cfs.sigma.max == pytest.approx(0.8 * (order + 1) / (z0 * DL))

    def test_inversely_proportional_to_cell_size(self, make_cfs):
        """Expects halving ``d`` to double ``sigma_max`` — a finer PML needs a
        steeper ramp to absorb as much over a shorter distance."""
        coarse, fine = make_cfs(), make_cfs()
        coarse.calculate_sigmamax(2 * DL, 1.0, 1.0)
        fine.calculate_sigmamax(DL, 1.0, 1.0)
        assert fine.sigma.max == pytest.approx(2 * coarse.sigma.max)

    def test_scales_with_one_over_root_er_mr(self, make_cfs):
        """Expects a PML backing ``er = 4`` to need half the conductivity of
        one backing free space, since ``sqrt(4·1) == 2``."""
        vacuum, soil = make_cfs(), make_cfs()
        vacuum.calculate_sigmamax(DL, 1.0, 1.0)
        soil.calculate_sigmamax(DL, 4.0, 1.0)
        assert soil.sigma.max == pytest.approx(vacuum.sigma.max / 2)

    def test_er_and_mr_enter_symmetrically(self, make_cfs):
        """Expects ``(er=4, mr=1)`` and ``(er=1, mr=4)`` to give the same
        answer — they appear only as the product under the root."""
        a, b = make_cfs(), make_cfs()
        a.calculate_sigmamax(DL, 4.0, 1.0)
        b.calculate_sigmamax(DL, 1.0, 4.0)
        assert a.sigma.max == pytest.approx(b.sigma.max)

    def test_writes_through_to_the_parameter(self, make_cfs):
        """Expects the result to be stored on ``sigma.max`` rather than
        returned — the caller relies on the mutation."""
        cfs = make_cfs()
        assert cfs.calculate_sigmamax(DL, 1.0, 1.0) is None
        assert cfs.sigma.max is not None


class TestScalingPolynomial:
    """The interleaved ``linspace``: even samples to E, odd samples to H."""

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_matches_an_independently_written_formula(self, make_cfs, order):
        """Expects agreement with a longhand reimplementation of the
        ``linspace``/stride construction. (3 parameter sets)"""
        cfs = make_cfs()
        zeros = np.zeros(5)
        E, H = cfs.scaling_polynomial(order, zeros, zeros)
        expE, expH = polynomial_profile(4, order)
        assert E == pytest.approx(expE)
        assert H == pytest.approx(expH)

    def test_linear_profile_samples_e_at_whole_cells(self, make_cfs):
        """Expects ``0, ¼, ½, ¾, 1`` for a linear ramp over five samples —
        the electric profile sits on cell boundaries."""
        cfs = make_cfs()
        E, _ = cfs.scaling_polynomial(1, np.zeros(5), np.zeros(5))
        assert E == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_linear_profile_samples_h_half_a_cell_later(self, make_cfs):
        """Expects ``⅛, ⅜, ⅝, ⅞, 1⅛`` — the magnetic profile trails the
        electric one by half a cell, which is the Yee stagger."""
        cfs = make_cfs()
        _, H = cfs.scaling_polynomial(1, np.zeros(5), np.zeros(5))
        assert H == pytest.approx([0.125, 0.375, 0.625, 0.875, 1.125])

    def test_the_two_profiles_interleave(self, make_cfs):
        """Expects every H sample to fall strictly between its neighbouring E
        samples — the defining property of a staggered pair."""
        cfs = make_cfs()
        E, H = cfs.scaling_polynomial(1, np.zeros(5), np.zeros(5))
        assert np.all(E[:-1] < H[:-1])
        assert np.all(H[:-1] < E[1:])

    def test_order_zero_is_flat(self, make_cfs):
        """Expects a constant profile to raise everything to the zeroth power,
        giving all ones — including the ``0 ** 0 == 1`` sample at the origin."""
        cfs = make_cfs()
        E, H = cfs.scaling_polynomial(0, np.zeros(5), np.zeros(5))
        assert E == pytest.approx(np.ones(5))
        assert H == pytest.approx(np.ones(5))

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_higher_orders_ramp_later(self, make_cfs, order):
        """Expects a steeper polynomial to hold the profile nearer zero in the
        interior — a quartic reaches ½ much later than a linear does.
        (4 parameter sets)"""
        cfs = make_cfs()
        E, _ = cfs.scaling_polynomial(order, np.zeros(5), np.zeros(5))
        assert E[2] == pytest.approx(0.5**order)

    def test_returns_new_arrays_rather_than_filling_the_inputs(self, make_cfs):
        """Expects the passed-in arrays to be untouched: the function returns
        replacements, and the arguments serve only to carry the length."""
        cfs = make_cfs()
        zeros = np.zeros(5)
        E, H = cfs.scaling_polynomial(1, zeros, zeros)
        assert E is not zeros and H is not zeros
        assert zeros == pytest.approx(np.zeros(5))


class TestCalculateValuesLength:
    @pytest.mark.parametrize("thickness", [1, 2, 4, 6, 10])
    def test_output_length_equals_thickness(self, make_cfs, thickness):
        """Expects one value per PML cell: the extra sample allocated to get
        the staggering right is dropped before returning. (5 parameter sets)"""
        cfs = make_cfs()
        E, H = cfs.calculate_values(thickness, cfs.kappa)
        assert len(E) == thickness
        assert len(H) == thickness

    def test_uses_the_configured_float_dtype(self, make_cfs):
        """Expects ``float64`` under the double-precision fixture — the arrays
        feed straight into Cython buffers typed by the same setting."""
        cfs = make_cfs()
        E, H = cfs.calculate_values(4, cfs.kappa)
        assert E.dtype == np.float64
        assert H.dtype == np.float64


class TestCalculateValuesConstant:
    """The ``constant`` profile short-circuits: it uses ``max`` directly and
    never consults ``min``."""

    def test_default_kappa_is_all_ones(self, make_cfs):
        """Expects ``[1, 1, 1, 1]`` — a stretch factor of one, i.e. off."""
        cfs = make_cfs()
        E, H = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx(np.ones(4))
        assert H == pytest.approx(np.ones(4))

    def test_default_alpha_is_all_zeros(self, make_cfs):
        """Expects ``[0, 0, 0, 0]`` — the frequency shift is off."""
        cfs = make_cfs()
        E, H = cfs.calculate_values(4, cfs.alpha)
        assert E == pytest.approx(np.zeros(4))
        assert H == pytest.approx(np.zeros(4))

    def test_constant_takes_max_not_min(self, make_cfs):
        """Expects every entry to equal ``max`` even when ``min`` differs —
        the constant branch fires before any min/max rescaling."""
        cfs = make_cfs(kappa={"min": 2.0, "max": 7.0})
        E, H = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx(np.full(4, 7.0))
        assert H == pytest.approx(np.full(4, 7.0))

    def test_electric_and_magnetic_agree(self, make_cfs):
        """Expects both profiles identical: a flat ramp has no stagger to
        express."""
        cfs = make_cfs(kappa={"max": 3.0})
        E, H = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx(H)

    def test_constant_profile_wins_over_the_scaling_field(self, make_cfs):
        """Expects ``scalingprofile == "constant"`` to be checked first, so it
        applies whatever ``scaling`` says."""
        cfs = make_cfs(kappa={"scaling": "something-else", "max": 5.0})
        E, _ = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx(np.full(4, 5.0))


class TestCalculateValuesPolynomial:
    """The polynomial branch, then rescaling into ``[min, max]``."""

    def test_linear_kappa_spans_min_to_max(self, make_cfs):
        """Expects ``min + (max-min)·t`` at the E sample points ``0, ¼, ½, ¾``
        — for ``min=1, max=5`` that is ``1, 2, 3, 4``."""
        cfs = make_cfs(kappa={"scalingprofile": "linear", "min": 1.0, "max": 5.0})
        E, _ = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_linear_kappa_magnetic_profile_is_offset(self, make_cfs):
        """Expects the H samples at ``⅛, ⅜, ⅝, ⅞`` rescaled the same way —
        ``1.5, 2.5, 3.5, 4.5``."""
        cfs = make_cfs(kappa={"scalingprofile": "linear", "min": 1.0, "max": 5.0})
        _, H = cfs.calculate_values(4, cfs.kappa)
        assert H == pytest.approx([1.5, 2.5, 3.5, 4.5])

    @pytest.mark.parametrize("name,order", [p for p in PROFILES if p[0] != "constant"])
    def test_every_polynomial_profile_matches_the_longhand_formula(self, make_cfs, name, order):
        """Expects agreement with an independently written reference for all
        eight non-constant profiles, rescaled into ``[0, 1]``. (8 parameter
        sets)"""
        cfs = make_cfs(sigma={"scalingprofile": name, "min": 0.0, "max": 1.0})
        E, H = cfs.calculate_values(4, cfs.sigma)
        expE, expH = polynomial_profile(4, order)
        assert E == pytest.approx(expE[:-1])
        assert H == pytest.approx(expH[:-1])

    def test_starts_at_min(self, make_cfs):
        """Expects the first electric sample to be exactly ``min``: the ramp
        begins at the inner face of the PML, where it must be invisible."""
        cfs = make_cfs(sigma={"scalingprofile": "quartic", "min": 0.25, "max": 9.0})
        E, _ = cfs.calculate_values(6, cfs.sigma)
        assert E[0] == pytest.approx(0.25)

    def test_is_monotonic(self, make_cfs):
        """Expects absorption to increase strictly outward — a non-monotonic
        ramp would create an internal reflection."""
        cfs = make_cfs(sigma={"scalingprofile": "quartic", "min": 0.0, "max": 10.0})
        E, H = cfs.calculate_values(8, cfs.sigma)
        assert np.all(np.diff(E) > 0)
        assert np.all(np.diff(H) > 0)

    def test_rescaling_is_selected_by_parameter_id(self, make_cfs):
        """Expects the min/max pair to be looked up by ``parameter.ID``, so a
        parameter whose ID does not match any of the three is left on the raw
        ``[0, 1]`` profile."""
        cfs = make_cfs()
        anonymous = CFSParameter(ID="not-a-cfs-term", scalingprofile="linear", min=3, max=9)
        E, _ = cfs.calculate_values(4, anonymous)
        assert E == pytest.approx([0.0, 0.25, 0.5, 0.75])

    def test_alpha_rescales_against_alpha(self, make_cfs):
        """Expects ``alpha``'s own min/max to be used, not sigma's — the three
        branches must not cross-wire."""
        cfs = make_cfs(
            alpha={"scalingprofile": "linear", "min": 0.0, "max": 4.0},
            sigma={"min": 100.0, "max": 200.0},
        )
        E, _ = cfs.calculate_values(4, cfs.alpha)
        assert E == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_sigma_rescales_against_sigma(self, make_cfs):
        """Expects ``sigma``'s own min/max to be used, mirroring the alpha
        case from the other side."""
        cfs = make_cfs(
            alpha={"min": 100.0, "max": 200.0},
            sigma={"scalingprofile": "linear", "min": 0.0, "max": 4.0},
        )
        E, _ = cfs.calculate_values(4, cfs.sigma)
        assert E == pytest.approx([0.0, 1.0, 2.0, 3.0])


class TestCalculateValuesReverse:
    """Slabs on the far side of the domain ramp the other way."""

    def test_electric_profile_is_the_forward_one_reversed(self, make_cfs):
        """Expects ``[1, ¾, ½, ¼]`` where forward gives ``[0, ¼, ½, ¾]`` —
        note the endpoints differ, because the dropped extra sample comes off
        the other end."""
        cfs = make_cfs(kappa={"scalingprofile": "linear", "min": 0.0, "max": 1.0})
        cfs.kappa.scalingdirection = "reverse"
        E, _ = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx([1.0, 0.75, 0.5, 0.25])

    def test_magnetic_profile_is_rolled_one_element_left(self, make_cfs):
        """Expects ``[⅞, ⅝, ⅜, ⅛]``. Reversing alone would give
        ``[1⅛, ⅞, ⅝, ⅜]``; the ``np.roll(-1)`` discards the out-of-range
        ``1⅛`` sample and restores the half-cell stagger."""
        cfs = make_cfs(kappa={"scalingprofile": "linear", "min": 0.0, "max": 1.0})
        cfs.kappa.scalingdirection = "reverse"
        _, H = cfs.calculate_values(4, cfs.kappa)
        assert H == pytest.approx([0.875, 0.625, 0.375, 0.125])

    def test_reverse_is_monotonically_decreasing(self, make_cfs):
        """Expects absorption to increase toward index 0 instead of away from
        it — the mirror image of the forward case."""
        cfs = make_cfs(sigma={"scalingprofile": "quartic", "min": 0.0, "max": 10.0})
        cfs.sigma.scalingdirection = "reverse"
        E, H = cfs.calculate_values(8, cfs.sigma)
        assert np.all(np.diff(E) < 0)
        assert np.all(np.diff(H) < 0)

    def test_h_still_falls_between_neighbouring_e_samples(self, make_cfs):
        """Expects the stagger to survive reversal — this is precisely what
        the roll exists to guarantee, and dropping it would leave every H
        value outside its E interval."""
        cfs = make_cfs(sigma={"scalingprofile": "linear", "min": 0.0, "max": 1.0})
        cfs.sigma.scalingdirection = "reverse"
        E, H = cfs.calculate_values(6, cfs.sigma)
        assert np.all(E[1:] < H[:-1])
        assert np.all(H[:-1] < E[:-1])

    def test_constant_profile_is_unaffected_by_reversal(self, make_cfs):
        """Expects a flat ramp reversed to still be flat — reversal is a
        no-op on a constant, which makes it a useful control."""
        cfs = make_cfs(kappa={"max": 3.0})
        cfs.kappa.scalingdirection = "reverse"
        E, H = cfs.calculate_values(4, cfs.kappa)
        assert E == pytest.approx(np.full(4, 3.0))
        assert H == pytest.approx(np.full(4, 3.0))

    def test_forward_and_reverse_electric_profiles_are_mirror_images(self, make_cfs):
        """Expects ``reverse(thickness+1 samples)[:-1]`` rather than
        ``forward[::-1]`` — the truncation happens after the reversal, so the
        two are not simply each other's ``[::-1]``."""
        fwd = make_cfs(sigma={"scalingprofile": "quadratic", "min": 0.0, "max": 1.0})
        rev = make_cfs(sigma={"scalingprofile": "quadratic", "min": 0.0, "max": 1.0})
        rev.sigma.scalingdirection = "reverse"
        E_fwd, _ = fwd.calculate_values(4, fwd.sigma)
        E_rev, _ = rev.calculate_values(4, rev.sigma)
        assert E_fwd != pytest.approx(E_rev[::-1])
        full, _ = polynomial_profile(4, 2)
        assert E_rev == pytest.approx(full[::-1][:-1])


pytestmark = pytest.mark.unit
