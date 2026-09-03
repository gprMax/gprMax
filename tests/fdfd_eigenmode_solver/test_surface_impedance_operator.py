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

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import (
    boundary_edge_relative_permittivity,
    evaluate_surface_ade,
)
from gprMax.impedance_surfaces import SurfaceImpedanceModel


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
                "c": 299792458.0,
                "z0": 376.73031366686166,
            }
        ),
    )


def test_surface_response_is_exact_bilinear_transfer_with_midpoint_admittance():
    model = SurfaceImpedanceModel(
        "first_order",
        A=((-2.0e9,),),
        B=(1.0e9,),
        C=(20.0,),
        D=30.0,
    )
    dt = 20e-12
    frequency = 2.2e9
    discrete = model.discretise(dt)

    response = evaluate_surface_ade(
        frequency_hz=frequency,
        dt=dt,
        F=discrete.F,
        G=discrete.G,
        L=discrete.L,
        Z0=discrete.Z0,
    )

    warped_s = 2j / dt * np.tan(response.theta / 2)
    expected_impedance = model.D + model.C @ np.linalg.solve(
        warped_s * np.eye(model.order) - model.A,
        model.B,
    )
    assert response.impedance == pytest.approx(expected_impedance, rel=2e-14, abs=2e-14)
    assert response.admittance == pytest.approx(
        np.cos(response.theta / 2) / expected_impedance,
        rel=2e-14,
        abs=2e-14,
    )


@pytest.mark.parametrize("discrete_normalization", (False, True))
def test_boundary_permittivity_reproduces_integral_discrete_ampere_load(discrete_normalization):
    epsilon0 = config.sim_config.em_consts["e0"]
    dt = 1e-12
    frequency = 20e9
    resistance = 40.0
    response = evaluate_surface_ade(
        frequency_hz=frequency,
        dt=dt,
        F=np.empty((0, 0)),
        G=np.empty(0),
        L=np.empty(0),
        Z0=resistance,
    )
    retained_area = 1.5e-6
    relative_permittivity = 3.2
    conductivity = 0.08
    electric_mass = epsilon0 * relative_permittivity * retained_area
    conductive_mass = conductivity * retained_area
    lengths = np.asarray((0.8e-3, 0.3e-3))
    normalization = (
        {"normalization_angular_frequency": response.discrete_angular_frequency}
        if discrete_normalization
        else {}
    )

    effective = boundary_edge_relative_permittivity(
        response=response,
        epsilon0=epsilon0,
        retained_dual_area=retained_area,
        electric_mass=electric_mass,
        conductive_mass=conductive_mass,
        port_lengths=lengths,
        **normalization,
    )

    normalization_frequency = (
        response.discrete_angular_frequency
        if discrete_normalization
        else response.physical_angular_frequency
    )
    represented_load = 1j * normalization_frequency * epsilon0 * retained_area * effective
    expected_load = (
        1j * response.discrete_angular_frequency * electric_mass
        + response.midpoint_cosine * conductive_mass
        + np.sum(lengths) * response.admittance
    )
    assert represented_load == pytest.approx(expected_load, rel=2e-14, abs=2e-14)
    assert response.physical_angular_frequency != response.discrete_angular_frequency


@pytest.mark.parametrize("normalization_frequency", (0.0, -1.0, np.nan, np.inf))
def test_boundary_permittivity_rejects_invalid_normalization_frequency(normalization_frequency):
    response = evaluate_surface_ade(
        frequency_hz=20e9,
        dt=1e-12,
        F=np.empty((0, 0)),
        G=np.empty(0),
        L=np.empty(0),
        Z0=40.0,
    )
    with pytest.raises(ValueError, match="normalization angular frequency.*finite and positive"):
        boundary_edge_relative_permittivity(
            response=response,
            epsilon0=config.sim_config.em_consts["e0"],
            retained_dual_area=1.5e-6,
            electric_mass=config.sim_config.em_consts["e0"] * 1.5e-6,
            conductive_mass=0.0,
            port_lengths=(0.8e-3,),
            normalization_angular_frequency=normalization_frequency,
        )


@pytest.mark.parametrize(
    "frequency, message",
    ((0.0, "positive"), (5.0e11, "Nyquist")),
)
def test_surface_response_rejects_unsupported_temporal_frequencies(frequency, message):
    with pytest.raises(ValueError, match=message):
        evaluate_surface_ade(
            frequency_hz=frequency,
            dt=1e-12,
            F=np.empty((0, 0)),
            G=np.empty(0),
            L=np.empty(0),
            Z0=1.0,
        )
