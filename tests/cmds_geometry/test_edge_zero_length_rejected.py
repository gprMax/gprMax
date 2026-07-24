"""Regression tests for rejecting zero-length electric edges."""
import pytest

import gprMax


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def test_edge_zero_length_rejected(tmp_path):
    scene = _scene()
    scene.add(gprMax.Edge(p1=(0.002, 0.002, 0.002), p2=(0.002, 0.002, 0.002), material_id="pec"))

    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True,
            outputfile=tmp_path / "edge_zero", hide_progress_bars=True,
        )


def test_edge_nonzero_length_still_accepted(tmp_path):
    """Sanity check the fix doesn't over-reach: a real, non-degenerate
    edge must still build without error."""
    scene = _scene()
    scene.add(gprMax.Edge(p1=(0.002, 0.001, 0.001), p2=(0.004, 0.001, 0.001), material_id="pec"))

    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "edge_nonzero", hide_progress_bars=True,
    )


def test_edge_zero_length_after_discretisation_rounding_rejected(tmp_path):
    """Two distinct continuous points that round to the SAME discretised
    grid coordinate are just as degenerate as literally equal p1/p2 -
    must be rejected too, not just the exact-equality case."""
    scene = _scene()
    scene.add(gprMax.Edge(p1=(0.002, 0.002, 0.002), p2=(0.0021, 0.002, 0.002), material_id="pec"))

    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True,
            outputfile=tmp_path / "edge_round_zero", hide_progress_bars=True,
        )
