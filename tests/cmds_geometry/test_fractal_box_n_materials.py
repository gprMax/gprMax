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

import pytest

import gprMax


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def test_zero_n_materials_rejected(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.FractalBox(
            p1=(0.01, 0.01, 0.01),
            p2=(0.02, 0.02, 0.02),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=0,
            mixing_model_id="pec",
            id="fb1",
        )
    )

    with pytest.raises(ValueError, match="positive value for the number of bins"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "zero_n_materials",
            hide_progress_bars=True,
        )


def test_positive_n_materials_still_works(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.SoilPeplinski(
            sand_fraction=0.5,
            clay_fraction=0.5,
            bulk_density=2.0,
            sand_density=2.66,
            water_fraction_lower=0.001,
            water_fraction_upper=0.25,
            id="soil1",
        )
    )
    scene.add(
        gprMax.FractalBox(
            p1=(0.01, 0.01, 0.01),
            p2=(0.02, 0.02, 0.02),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=2,
            mixing_model_id="soil1",
            id="fb1",
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "positive_n_materials",
        hide_progress_bars=True,
    )
