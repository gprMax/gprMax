"""Regression tests for fractal modifier hash-command validation."""

import pytest

import gprMax
from gprMax.hash_cmds_geometry import process_geometrycmds


@pytest.mark.parametrize(
    "command",
    (
        "#add_surface_roughness: 0 0 0 1 1 0 1.5 1 1 0 1 absent",
        "#add_surface_water: 0 0 0 1 1 0 0.1 absent",
        "#add_grass: 0 0 0 1 1 0 1.5 0.01 0.02 10 absent",
    ),
)
def test_orphaned_fractal_modifier_is_not_silently_discarded(command):
    with pytest.raises(ValueError, match="cannot find #fractal_box.*absent"):
        process_geometrycmds([command])


@pytest.mark.parametrize(
    "command",
    (
        "#add_surface_roughness: 0 0 0",
        "#add_surface_water: 0 0 0",
        "#add_grass: 0 0 0",
    ),
)
def test_malformed_fractal_modifier_is_rejected_without_a_fractal_box(command):
    with pytest.raises(ValueError, match="requires exactly"):
        process_geometrycmds([command])


def test_duplicate_fractal_box_identifiers_are_rejected():
    box = "#fractal_box: 0 0 0 1 1 1 1.5 1 1 1 1 soil duplicate"

    with pytest.raises(ValueError, match="identifiers must be unique"):
        process_geometrycmds([box, box])


def test_python_api_duplicate_fractal_box_identifiers_are_rejected(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Material(er=2, se=0, mr=1, sm=0, id="soil"))
    for offset in (0.0, 0.02):
        scene.add(
            gprMax.FractalBox(
                p1=(offset, 0, 0),
                p2=(offset + 0.02, 0.02, 0.02),
                frac_dim=1.5,
                weighting=(1, 1, 1),
                n_materials=1,
                mixing_model_id="soil",
                id="duplicate",
                seed=1,
            )
        )

    with pytest.raises(ValueError, match="already in use"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "duplicate_fractal_id",
            hide_progress_bars=True,
        )
