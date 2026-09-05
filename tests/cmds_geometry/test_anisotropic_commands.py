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

"""Regression coverage for anisotropic geometry commands and material reuse."""

import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.hash_cmds_geometry import process_geometrycmds
from gprMax.materials import Material
from gprMax.user_objects.cmds_geometry.ellipsoid import Ellipsoid
from gprMax.user_objects.cmds_geometry.sphere import Sphere

MATERIAL_IDS = ("mat_x", "mat_y", "mat_z")


@pytest.mark.parametrize(
    "command, expected_type",
    (
        ("#sphere: 0.01 0.01 0.01 0.004 mat_x mat_y mat_z", Sphere),
        ("#ellipsoid: 0.01 0.01 0.01 0.004 0.003 0.002 mat_x mat_y mat_z", Ellipsoid),
    ),
)
def test_anisotropic_sphere_and_ellipsoid_hash_commands(command, expected_type):
    objects = process_geometrycmds([command])

    assert len(objects) == 1
    assert isinstance(objects[0], expected_type)
    assert objects[0].kwargs["material_ids"] == list(MATERIAL_IDS)


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Material(er=2, se=0, mr=1, sm=0, id=MATERIAL_IDS[0]))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id=MATERIAL_IDS[1]))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id=MATERIAL_IDS[2]))

    # Seed the reusable compound material before the primitive under test.
    scene.add(
        gprMax.Box(
            p1=(0.002, 0.002, 0.002),
            p2=(0.006, 0.006, 0.006),
            material_ids=MATERIAL_IDS,
        )
    )
    return scene


def _box():
    return gprMax.Box(
        p1=(0.010, 0.010, 0.010),
        p2=(0.016, 0.016, 0.016),
        material_ids=MATERIAL_IDS,
    )


def _sphere():
    return gprMax.Sphere(p1=(0.014, 0.014, 0.014), r=0.004, material_ids=MATERIAL_IDS)


def _ellipsoid():
    return gprMax.Ellipsoid(
        p1=(0.014, 0.014, 0.014),
        xr=0.004,
        yr=0.003,
        zr=0.002,
        material_ids=MATERIAL_IDS,
    )


def _cylinder():
    return gprMax.Cylinder(
        p1=(0.010, 0.014, 0.014),
        p2=(0.018, 0.014, 0.014),
        r=0.003,
        material_ids=MATERIAL_IDS,
    )


def _cone():
    return gprMax.Cone(
        p1=(0.010, 0.014, 0.014),
        p2=(0.018, 0.014, 0.014),
        r1=0.003,
        r2=0.002,
        material_ids=MATERIAL_IDS,
    )


def _cylindrical_sector():
    return gprMax.CylindricalSector(
        normal="z",
        ctr1=0.014,
        ctr2=0.014,
        extent1=0.010,
        extent2=0.018,
        r=0.004,
        start=0,
        end=90,
        material_ids=MATERIAL_IDS,
    )


def _triangle():
    return gprMax.Triangle(
        p1=(0.010, 0.010, 0.010),
        p2=(0.018, 0.010, 0.010),
        p3=(0.010, 0.018, 0.010),
        thickness=0.004,
        material_ids=MATERIAL_IDS,
    )


@pytest.mark.parametrize(
    "factory",
    (_box, _sphere, _ellipsoid, _cylinder, _cone, _cylindrical_sector, _triangle),
)
def test_anisotropic_primitives_reuse_existing_compound_material(
    factory,
    monkeypatch,
    tmp_path,
):
    scene = _base_scene()
    scene.add(factory())
    captured = _capture_grid(monkeypatch)

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / factory.__name__,
        hide_progress_bars=True,
    )

    grid = captured["grid"]
    constituents = [
        next(material for material in grid.materials if material.ID == material_id)
        for material_id in MATERIAL_IDS
    ]
    compound_id = Material.create_compound_id(*constituents)
    assert sum(material.ID == compound_id for material in grid.materials) == 1


def test_missing_anisotropic_material_error_names_only_missing_id(monkeypatch, tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.010, 0.010, 0.010),
            p2=(0.016, 0.016, 0.016),
            material_ids=(MATERIAL_IDS[0], "missing", MATERIAL_IDS[2]),
        )
    )
    _capture_grid(monkeypatch)

    with pytest.raises(ValueError, match=r"material\(s\) \['missing'\] do not exist"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "missing_material",
            hide_progress_bars=True,
        )


@pytest.mark.parametrize(
    "geometry,expected_count",
    (
        (
            gprMax.Box(
                p1=(0.008, 0.008, 0.008),
                p2=(0.016, 0.016, 0.016),
                material_ids=MATERIAL_IDS[:2],
            ),
            3,
        ),
        (
            gprMax.Plate(
                p1=(0.008, 0.008, 0.012),
                p2=(0.016, 0.016, 0.012),
                material_ids=MATERIAL_IDS,
            ),
            2,
        ),
    ),
)
def test_invalid_directional_material_count_is_reported(geometry, expected_count, tmp_path):
    scene = _base_scene()
    scene.add(geometry)

    with pytest.raises(ValueError, match=rf"requires exactly {expected_count}"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / f"count_{expected_count}",
            hide_progress_bars=True,
        )


def test_anisotropic_surface_sector_accepts_typed_material_id(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.CylindricalSector(
            normal="z",
            ctr1=0.014,
            ctr2=0.014,
            extent1=0.012,
            extent2=0.012,
            r=0.004,
            start=0,
            end=90,
            material_ids=MATERIAL_IDS,
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "anisotropic_surface_sector",
        hide_progress_bars=True,
    )
