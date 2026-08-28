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

import gprMax


def test_thickness_geometry_builds_in_off_origin_subgrid(tmp_path):
    """Triangles and sectors must not validate placeholder transverse zeros."""
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))

    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    subgrid.add(
        gprMax.Triangle(
            p1=(0.040, 0.040, 0.045),
            p2=(0.050, 0.040, 0.045),
            p3=(0.045, 0.052, 0.045),
            thickness=0.002,
            material_id="pec",
        )
    )
    subgrid.add(
        gprMax.CylindricalSector(
            normal="z",
            ctr1=0.045,
            ctr2=0.045,
            extent1=0.048,
            extent2=0.051,
            r=0.004,
            start=0,
            end=180,
            material_id="pec",
        )
    )
    scene.add(subgrid)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "off_origin_thickness_geometry",
        geometry_only=True,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    pec = next(material for material in subgrid.subgrid.materials if material.ID == "pec")
    assert np.count_nonzero(subgrid.subgrid.solid == pec.numID) > 0
