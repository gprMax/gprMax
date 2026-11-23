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

"""End-to-end regression tests for degenerate (single-cell) fractal-surface
footprints, building real models through ``gprMax.run(geometry_only=True)``.

Mirrors the harness of ``tests/fractals/test_fractal_te_invariance.py`` and
``tests/cmds_geometry/test_degenerate_primitives.py``. A footprint that
discretises to a single cell in both in-plane directions used to crash with
``FloatingPointError: divide by zero encountered in divide`` from
``fractal_surface.py``; it must now raise a clean ``ValueError`` naming the
minimum size, while every footprint with at least 2 cells in one in-plane
direction (2x2, 1xN and Nx1 strips) still builds.
"""

import pytest

import gprMax

DL = 1e-3


def _scene(p2, mode=None):
    scene = gprMax.Scene()
    if mode is not None:
        scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Title(name="rough1x1"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(6 * DL, 6 * DL, 6 * DL)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="soilmat"))
    scene.add(
        gprMax.FractalBox(
            p1=(0, 0, 0),
            p2=(3 * DL, 3 * DL, 4 * DL),
            frac_dim=2.5,
            weighting=(1, 1, 1),
            n_materials=1,
            mixing_model_id="soilmat",
            id="soil",
            seed=1,
        )
    )
    scene.add(
        gprMax.AddSurfaceRoughness(
            p1=(0, 0, 4 * DL),
            p2=p2,
            frac_dim=2.5,
            weighting=(1, 1),
            limits=(2 * DL, 6 * DL),
            fractal_box_id="soil",
            seed=2,
        )
    )
    return scene


def _run(scene, tmp_path, name):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
    )


def test_1x1_footprint_raises_a_clean_value_error(tmp_path):
    with pytest.raises(ValueError, match="at least 2 cells"):
        _run(_scene((DL, DL, 4 * DL)), tmp_path, "rough_1x1")


def test_sub_cell_footprint_that_rounds_down_to_1x1_raises(tmp_path):
    # 1.4 cells in both in-plane directions discretises down to 1 x 1.
    with pytest.raises(ValueError, match="at least 2 cells"):
        _run(_scene((1.4 * DL, 1.4 * DL, 4 * DL)), tmp_path, "rough_14")


@pytest.mark.parametrize(
    "p2, name",
    [
        ((2 * DL, 2 * DL, 4 * DL), "rough_2x2"),
        ((DL, 3 * DL, 4 * DL), "rough_1x3"),
        ((3 * DL, DL, 4 * DL), "rough_3x1"),
        ((3 * DL, 3 * DL, 4 * DL), "rough_3x3"),
        ((1.6 * DL, 1.6 * DL, 4 * DL), "rough_16"),  # rounds up to 2 x 2
    ],
)
def test_control_footprints_still_build(tmp_path, p2, name):
    _run(_scene(p2), tmp_path, name)


def test_grass_1x1_footprint_raises_a_clean_value_error(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="grass1x1"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(6 * DL, 6 * DL, 6 * DL)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="soilmat"))
    scene.add(
        gprMax.FractalBox(
            p1=(0, 0, 0),
            p2=(3 * DL, 3 * DL, 4 * DL),
            frac_dim=2.5,
            weighting=(1, 1, 1),
            n_materials=1,
            mixing_model_id="soilmat",
            id="soil",
            seed=1,
        )
    )
    scene.add(
        gprMax.AddGrass(
            p1=(0, 0, 4 * DL),
            p2=(DL, DL, 4 * DL),
            frac_dim=2.5,
            limits=(5 * DL, 6 * DL),
            n_blades=1,
            fractal_box_id="soil",
            seed=3,
        )
    )
    with pytest.raises(ValueError, match="at least 2 cells"):
        _run(scene, tmp_path, "grass_1x1")
