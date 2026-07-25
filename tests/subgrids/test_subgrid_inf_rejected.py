"""Regression test: `inf` coordinates must be rejected for subgrid-scoped
objects, not silently mis-resolved.

The resolve_inf_point() "axis origin = 0" /
"axis extent = nx*dl" formula assumes the grid's own coordinate frame
starts at 0 - true for the main grid, but false for a subgrid, whose
commands take coordinates in the *global* (main-grid) frame. A subgrid
Box with `inf` either crashed (out-of-bounds local index) or - worse -
silently landed at a wrong-but-in-bounds position (confirmed: a
HertzianDipole's `inf` resolved to global x=0.066, landing inside the
subgrid's own boundary/PML padding, even though the subgrid's physical
footprint started at global x=0.075 - no error, just a wrong answer).

Fix: `inf` is now rejected outright unless the model is in an active 2D
mode (TM/TE). Since 2D mode and sub-gridding are already mutually
exclusive (enforced in Domain.build()), a model with any subgrid is
always "3D" from resolve_inf_point()'s point of view, so this blocks
subgrid `inf` usage for free, with no subgrid-specific code needed.
"""
import tempfile
from pathlib import Path

import pytest

import gprMax

INF = float("inf")


def _subgrid_scene():
    ratio = 3
    dl_sg = 1e-3
    dl_main = dl_sg * ratio

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl_main, dl_main, dl_main)))
    scene.add(gprMax.Domain(p1=(0.18, 0.18, 0.18)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    subgrid = gprMax.SubGridHSG(
        p1=(0.075, 0.075, 0.075), p2=(0.105, 0.105, 0.105), ratio=ratio, id="sg"
    )
    scene.add(subgrid)
    return scene, subgrid


def test_subgrid_box_with_inf_is_rejected(tmp_path):
    scene, subgrid = _subgrid_scene()
    subgrid.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    subgrid.add(gprMax.Box(p1=(INF, INF, INF), p2=(INF, INF, INF), material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "sg_box_inf",
            subgrid=True,
            autotranslate=True,
            hide_progress_bars=True,
        )


def test_subgrid_source_with_inf_is_rejected(tmp_path):
    scene, subgrid = _subgrid_scene()
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="mypulse"))
    subgrid.add(
        gprMax.HertzianDipole(polarisation="z", p1=(INF, 0.09, 0.09), waveform_id="mypulse")
    )

    with pytest.raises(ValueError, match="2D"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "sg_src_inf",
            subgrid=True,
            autotranslate=True,
            hide_progress_bars=True,
        )
