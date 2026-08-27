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

"""Perfect electric and magnetic conductors cannot be made dispersive."""

import pytest

import gprMax


def _dispersion(formulation, material_id):
    common = {"poles": 1, "material_ids": [material_id]}
    if formulation == "debye":
        return gprMax.AddDebyeDispersion(
            er_delta=[2.0],
            tau=[1e-10],
            **common,
        )
    if formulation == "lorentz":
        return gprMax.AddLorentzDispersion(
            er_delta=[2.0],
            omega=[1e9],
            delta=[1e8],
            **common,
        )
    return gprMax.AddDrudeDispersion(
        omega=[1e9],
        alpha=[1e8],
        **common,
    )


@pytest.mark.parametrize("formulation", ("debye", "lorentz", "drude"))
@pytest.mark.parametrize(
    ("material_id", "material_kwargs", "conductor_type"),
    (
        ("pec", None, "PEC"),
        ("pmc", None, "PMC"),
        ("custom_pec", {"er": 1, "se": float("inf"), "mr": 1, "sm": 0}, "PEC"),
        ("custom_pmc", {"er": 1, "se": 0, "mr": 1, "sm": float("inf")}, "PMC"),
    ),
)
def test_dispersion_rejects_perfect_conductors(
    formulation,
    material_id,
    material_kwargs,
    conductor_type,
    tmp_path,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(4e-3, 4e-3, 4e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    if material_kwargs is not None:
        scene.add(gprMax.Material(id=material_id, **material_kwargs))
    scene.add(_dispersion(formulation, material_id))

    message = rf"cannot add {formulation.title()} electric dispersion.*{conductor_type}"
    with pytest.raises(ValueError, match=message):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / f"{formulation}_{material_id}",
            hide_progress_bars=True,
        )
