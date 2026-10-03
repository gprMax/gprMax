# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest

from gprMax.toolboxes.DebyeFit.Debye_Fit import Crim, HavriliakNegami
from gprMax.toolboxes.DebyeFit.optimization import DLS, PSO_DLS

pytestmark = pytest.mark.unit


def test_crim_calculation_broadcasts_volumetric_fractions_per_frequency_row():
    """Regression test for a bug where Crim.calculation() built the
    per-frequency fractions matrix via
    ``np.repeat(volumetric_fractions, len(freq)).reshape((-1, len(materials)))``,
    which does NOT broadcast [f0, f1, f2] to every frequency row - it
    produces blocks of constant-fraction rows (e.g. f0 repeated for the
    first third of frequency points, f1 for the next third, ...), silently
    scrambling every CRIM fit that used more than one material. Fixed by
    relying on plain numpy broadcasting instead.
    """
    fractions = np.array([0.6, 0.119, 0.281])
    materials = np.array([[5.0, 0.0, 1.0], [4.9, 73.34, 8.0994e-12], [1.0, 0.0, 1.0]])

    crim = Crim(
        f_min=1e6,
        f_max=3e9,
        a=0.5,
        volumetric_fractions=fractions,
        materials=materials,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="regression_test",
        f_n=60,
    )
    result = crim.calculation()

    # Exact CRIM value at the lowest (near-static) frequency, computed
    # independently of Crim.calculation()'s internal array machinery.
    w0 = 2 * np.pi * crim.freq[0]
    eps_water_static = 4.9 + 73.34 / (1 + 1j * w0 * 8.0994e-12)
    expected = (0.6 * 5.0**0.5 + 0.119 * eps_water_static**0.5 + 0.281 * 1.0**0.5) ** (
        1 / 0.5
    )

    assert result[0] == pytest.approx(expected, rel=1e-9)


def _crim_with_fractions(fractions):
    materials = np.array([[5.0, 0.0, 1.0], [4.9, 73.34, 8.0994e-12], [1.0, 0.0, 1.0]])
    return Crim(
        f_min=1e6,
        f_max=3e9,
        a=0.5,
        volumetric_fractions=fractions,
        materials=materials[: len(fractions)],
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="crim_fraction_sum",
        number_of_debye_poles=1,
        f_n=10,
    )


@pytest.mark.parametrize("fractions", [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
def test_crim_accepts_fractions_summing_to_one_with_rounding_error(fractions):
    _crim_with_fractions(fractions).check_inputs()


@pytest.mark.parametrize("fractions", [[0.7, 0.2, 0.2], [0.5, 0.3, 0.1]])
def test_crim_rejects_fractions_not_summing_to_one(fractions):
    with pytest.raises(ValueError, match="summation of volumetric volumes"):
        _crim_with_fractions(fractions).check_inputs()


def test_imaginary_part_error_is_normalised_by_loss_magnitude():
    model = HavriliakNegami(
        f_min=1e6,
        f_max=1e9,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="imag_error",
        number_of_debye_poles=1,
        f_n=3,
    )
    model.rl = np.full(3, 4.0)
    model.im = np.array([-0.5, -1.0, -2.0])

    err_real, err_imag = model.error(model.rl, model.im + 0.01)

    assert err_real == 0
    assert err_imag == pytest.approx(np.mean(0.01 / np.array([1.5, 2.0, 3.0])) * 100)


def test_dls_constrains_infinite_frequency_permittivity_to_unity_or_greater():
    """The fitted infinite-frequency relative permittivity must not be less
    than that of vacuum."""
    freq = np.logspace(6, 9, 50)
    tt = np.array([-10.0])  # log10(tau), arbitrary single pole

    # A deliberately sub-unity target exercises the physical lower bound.
    rl = np.full_like(freq, 0.5)
    im = np.zeros_like(freq)

    with pytest.warns(UserWarning, match="physical lower bound of 1"):
        _, _, _, ee, _, _ = DLS(tt, rl, im, freq)

    assert ee == pytest.approx(1.0)


def test_dls_reports_severely_invalid_unconstrained_fit():
    freq = np.logspace(6, 9, 20)

    with pytest.warns(UserWarning, match=r"-100.*physical lower bound"):
        _, _, _, ee, _, _ = DLS(
            np.array([-10.0]),
            np.full_like(freq, -100.0),
            np.zeros_like(freq),
            freq,
        )

    assert ee == pytest.approx(1.0)


def test_auto_pole_count_matches_the_number_of_poles_in_the_accepted_fit(monkeypatch):
    """Regression test for a bug where Relaxation.run()'s automatic
    pole-count search (``number_of_debye_poles=-1``) incremented
    ``self.number_of_debye_poles`` unconditionally at the end of every
    loop iteration, including the one that met the error threshold and
    broke the loop - leaving ``self.number_of_debye_poles`` one higher
    than the pole count actually used to produce the returned fit.
    """
    model = HavriliakNegami(
        f_min=1e6,
        f_max=1e9,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="auto_pole_count_test",
        number_of_debye_poles=-1,
        f_n=12,
        plot=False,
        save=False,
        optimizer=PSO_DLS,
    )

    def accepted_one_pole_fit():
        size = model.number_of_debye_poles
        tau = np.full(size, 1e-9)
        weights = np.full(size, 5.0 / size)
        return tau, weights, 3.0, model.rl - 3.0, model.im

    monkeypatch.setattr(model, "optimize", accepted_one_pole_fit)

    _, properties = model.run()
    n_poles_in_output = int(properties[1].split()[1])

    assert model.number_of_debye_poles == n_poles_in_output


def test_short_havriliak_negami_fit_produces_gprmax_material_commands():
    model = HavriliakNegami(
        f_min=1e6,
        f_max=1e9,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="smoke_test",
        number_of_debye_poles=1,
        f_n=12,
        plot=False,
        save=False,
        optimizer_options={"swarmsize": 4, "maxiter": 2, "seed": 1},
    )

    error, properties = model.run()

    assert np.isfinite(error)
    assert any(line.startswith("#material:") for line in properties)
    assert any(line.startswith("#add_dispersion_debye:") for line in properties)


def test_debye_fit_and_optimizer_defaults_not_mutable():
    import inspect

    from gprMax.toolboxes.DebyeFit import Debye_Fit, optimization

    classes = [
        Debye_Fit.Relaxation,
        Debye_Fit.HavriliakNegami,
        Debye_Fit.Jonscher,
        Debye_Fit.Crim,
        Debye_Fit.Rawdata,
    ]
    for cls in classes:
        sig = inspect.signature(cls.__init__)
        assert sig.parameters["optimizer_options"].default is None

    optimizer_classes = [
        optimization.PSO_DLS,
        optimization.DA_DLS,
        optimization.DE_DLS,
    ]
    assert inspect.signature(optimization.Optimizer.fit).parameters["funckwargs"].default is None
    assert (
        inspect.signature(optimization.DA_DLS.__init__).parameters["local_search_options"].default
        is None
    )
    for cls in optimizer_classes:
        sig = inspect.signature(cls.calc_relaxation_times)
        assert sig.parameters["funckwargs"].default is None
