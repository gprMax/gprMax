# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Analytical-permittivity regressions for Lorentz and Drude materials."""

import numpy as np

import gprMax
import gprMax.config as config
import gprMax.model as model_module
from gprMax.cython.fields_updates_dispersive import (
    update_electric_dispersive_1pole_A_double_complex,
)
from gprMax.materials import DispersiveMaterial


def _material(kind):
    material = DispersiveMaterial(0, kind)
    material.type = kind
    material.er = 3.25
    material.se = 0
    material.poles = 2
    material.tau = [1.2e9, 3.2e9]
    material.alpha = [0.4e9, 0.8e9]
    material.deltaer = [1.5, 0.8]
    return material


def test_lorentz_calculate_er_uses_pole_frequencies_in_hertz():
    material = _material("lorentz")
    frequency = 2e9
    omega = 2 * np.pi * frequency
    expected = complex(material.er)
    for delta_er, pole_frequency, damping in zip(
        material.deltaer, material.tau, material.alpha
    ):
        pole_omega = 2 * np.pi * pole_frequency
        expected += delta_er * pole_omega**2 / (
            pole_omega**2 + 2j * omega * damping - omega**2
        )

    np.testing.assert_allclose(material.calculate_er(frequency), expected)


def test_multipole_drude_calculate_er_converts_and_sums_each_pole_once():
    material = _material("drude")
    frequency = 2e9
    omega = 2 * np.pi * frequency
    expected = complex(material.er)
    for pole_frequency, damping in zip(material.tau, material.alpha):
        pole_omega = 2 * np.pi * pole_frequency
        expected -= pole_omega**2 / (omega**2 - 1j * omega * damping)

    np.testing.assert_allclose(material.calculate_er(frequency), expected)


def test_complex_pole_current_uses_real_part_of_complete_product():
    """The Lorentz imaginary cross term must contribute to apparent current."""

    nx, ny, nz = 1, 2, 2
    shape = (nx + 1, ny + 1, nz + 1)
    updatecoeffs_e = np.zeros((1, 5), dtype=np.float64)
    updatecoeffs_e[0, 0] = 1
    updatecoeffs_e[0, 4] = 1
    updatecoeffs_dispersive = np.zeros((1, 3), dtype=np.complex128)
    updatecoeffs_dispersive[0] = (1 + 2j, 1, 0)
    material_ids = np.zeros((6, *shape), dtype=np.uint32)
    pole_shape = (1, *shape)
    tx = np.zeros(pole_shape, dtype=np.complex128)
    ty = np.zeros_like(tx)
    tz = np.zeros_like(tx)
    tx[0, 0, 1, 1] = 3 + 4j
    ex = np.zeros(shape, dtype=np.float64)
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    hx = np.zeros_like(ex)
    hy = np.zeros_like(ex)
    hz = np.zeros_like(ex)

    update_electric_dispersive_1pole_A_double_complex(
        nx,
        ny,
        nz,
        0,
        1,
        1,
        updatecoeffs_e,
        updatecoeffs_dispersive,
        material_ids,
        tx,
        ty,
        tz,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
    )

    # Re((1 + 2j) * (3 + 4j)) = -5, so E_new = -phi = +5.
    # The former Re(a) * Re(T) implementation incorrectly produced -3.
    assert ex[0, 1, 1] == 5


def test_drude_update_uses_dimensioned_effective_conductivity_without_mutation(
    monkeypatch, tmp_path
):
    captured = {}
    original_build = model_module.Model.build

    def capture_build(model):
        original_build(model)
        captured["grid"] = model.G

    monkeypatch.setattr(model_module.Model, "build", capture_build)

    physical_conductivity = 0.125
    pole_frequencies = (1e9, 2e9)
    damping = (0.8e9, 1.2e9)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(10e-3,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(
        gprMax.Material(
            er=3.25,
            se=physical_conductivity,
            mr=1,
            sm=0,
            id="drude",
        )
    )
    scene.add(
        gprMax.AddDrudeDispersion(
            poles=2,
            omega=pole_frequencies,
            alpha=damping,
            material_ids=["drude"],
        )
    )
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "drude_coefficients",
        hide_progress_bars=True,
    )

    grid = captured["grid"]
    material = next(item for item in grid.materials if item.ID == "drude")
    assert material.se == physical_conductivity

    e0 = config.e0
    effective_conductivity = physical_conductivity + e0 * sum(
        (2 * np.pi * frequency) ** 2 / rate
        for frequency, rate in zip(pole_frequencies, damping)
    )
    ea = (
        e0 * material.er / grid.dt
        + 0.5 * effective_conductivity
        - e0 / grid.dt * np.sum(material.zt2.real)
    )
    eb = (
        e0 * material.er / grid.dt
        - 0.5 * effective_conductivity
        - e0 / grid.dt * np.sum(material.zt2.real)
    )
    np.testing.assert_allclose(material.CA, eb / ea)
    np.testing.assert_allclose(material.srce, 1 / ea)
