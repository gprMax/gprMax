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


def test_dls_does_not_clamp_a_legitimately_sub_unity_real_part_offset():
    """Regression test for a bug where DLS() (optimization.py) computed
    ``ee = mean(rl - rp)`` (the fitted real-part offset, written directly
    into the output ``#material:`` line) and then silently clamped it with
    ``ee = max(ee, 1)``. For any target curve whose true residual is
    legitimately below 1, this replaced the correct value with a hard 1.0
    with no warning. Fixed by removing the clamp - any resulting
    unphysical material (er < 1) is now caught by gprMax's own
    ``#material`` validation instead of being silently masked here.
    """
    freq = np.logspace(6, 9, 50)
    tt = np.array([-10.0])  # log10(tau), arbitrary single pole

    # Flat, lossless target with a real part of 0.5 - the correct fitted
    # weight is ~0 and the correct ee is ~0.5, well below the old clamp.
    rl = np.full_like(freq, 0.5)
    im = np.zeros_like(freq)

    _, _, _, ee, _, _ = DLS(tt, rl, im, freq)

    assert ee == pytest.approx(0.5, abs=1e-9)


def test_run_warns_instead_of_silently_masking_a_sub_unity_e_inf():
    """A relative permittivity at infinite frequency below 1 (vacuum) is not
    physically valid for a passive dielectric - gprMax's own #material
    command rejects it. The old ``max(ee, 1)`` clamp in DLS() tried to
    enforce this but did so by silently overwriting the fitted value with
    exactly 1.0, whether the true fit wanted 0.8 (a mild, arguably
    reasonable result) or -10.6 (a badly non-converged fit) - both were
    made to look identically "valid" with no indication anything was
    wrong. Fixed by keeping the true fitted value and raising a warning
    instead, so a bad fit is visible rather than silently laundered.
    """
    model = HavriliakNegami(
        f_min=1e7,
        f_max=1e11,
        alpha=0.91,
        beta=1.0,
        e_inf=0.8,
        de=3.0,
        tau_0=9.4e-10,
        sigma=0,
        mu=0,
        mu_sigma=0,
        material_name="sub_unity_probe",
        number_of_debye_poles=3,
        f_n=30,
        plot=False,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options={"swarmsize": 15, "maxiter": 15, "seed": 1},
    )

    with pytest.warns(UserWarning, match="less than the permittivity of vacuum"):
        _, properties = model.run()

    fitted_er = float(properties[0].split()[1])
    # The true fitted value must survive - not be silently forced to 1.0.
    assert fitted_er < 1
    assert fitted_er != 1.0


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
