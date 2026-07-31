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
