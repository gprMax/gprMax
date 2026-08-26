# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

import numpy as np
import pytest

from toolboxes.DebyeFit.Debye_Fit import Crim, HavriliakNegami
from toolboxes.DebyeFit.optimization import DLS, PSO_DLS


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
