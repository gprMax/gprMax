"""Regression tests for geometry definitions that collapse after gridding."""

import pytest

import gprMax


def _scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(iterations=1))
    return scene


@pytest.mark.parametrize(
    "name,geometry,error",
    [
        (
            "sphere",
            gprMax.Sphere(p1=(0.01, 0.01, 0.01), r=0, material_id="pec"),
            "radius",
        ),
        (
            "ellipsoid",
            gprMax.Ellipsoid(
                p1=(0.01, 0.01, 0.01),
                xr=0.003,
                yr=0,
                zr=0.003,
                material_id="pec",
            ),
            "semiaxes",
        ),
        (
            "cylinder",
            gprMax.Cylinder(
                p1=(0.008, 0.008, 0.008),
                p2=(0.0084, 0.008, 0.008),
                r=0.002,
                material_id="pec",
            ),
            "different grid points",
        ),
        (
            "cone",
            gprMax.Cone(
                p1=(0.008, 0.008, 0.008),
                p2=(0.0084, 0.008, 0.008),
                r1=0.001,
                r2=0.002,
                material_id="pec",
            ),
            "different grid points",
        ),
        (
            "sector",
            gprMax.CylindricalSector(
                normal="z",
                ctr1=0.01,
                ctr2=0.01,
                extent1=0.007,
                extent2=0.012,
                r=0,
                start=0,
                end=90,
                material_id="pec",
            ),
            "radius",
        ),
        (
            "triangle",
            gprMax.Triangle(
                p1=(0.005, 0.005, 0.01),
                p2=(0.008, 0.008, 0.01),
                p3=(0.011, 0.011, 0.01),
                thickness=0.002,
                material_id="pec",
            ),
            "non-degenerate triangle",
        ),
        (
            "sphere_nan",
            gprMax.Sphere(p1=(0.01, 0.01, 0.01), r=float("nan"), material_id="pec"),
            "positive value",
        ),
        (
            "ellipsoid_inf",
            gprMax.Ellipsoid(
                p1=(0.01, 0.01, 0.01),
                xr=0.003,
                yr=float("inf"),
                zr=0.003,
                material_id="pec",
            ),
            "semiaxes",
        ),
        (
            "cylinder_nan",
            gprMax.Cylinder(
                p1=(0.008, 0.008, 0.008),
                p2=(0.012, 0.008, 0.008),
                r=float("nan"),
                material_id="pec",
            ),
            "positive value",
        ),
        (
            "cone_inf",
            gprMax.Cone(
                p1=(0.008, 0.008, 0.008),
                p2=(0.012, 0.008, 0.008),
                r1=0.001,
                r2=float("inf"),
                material_id="pec",
            ),
            "positive value",
        ),
        (
            "sector_nan",
            gprMax.CylindricalSector(
                normal="z",
                ctr1=0.01,
                ctr2=0.01,
                extent1=0.007,
                extent2=0.012,
                r=0.002,
                start=float("nan"),
                end=90,
                material_id="pec",
            ),
            "finite",
        ),
    ],
)
def test_degenerate_geometry_is_rejected(name, geometry, error, tmp_path):
    scene = _scene()
    scene.add(geometry)

    with pytest.raises(ValueError, match=error):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / name,
            hide_progress_bars=True,
        )
