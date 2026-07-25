"""End-to-end tests for `inf` coordinates in #geometry_view / GeometryView,
covering the same range-endpoint resolution rules already proven for Box/
Snapshot/GeometryObjectsWrite (gprMax/user_inputs.py's resolve_inf_point(),
role="lower"/"upper"): purely positional, spans the full invariant-axis
thickness in TM/TE 2D mode with no special-casing, and rejected outright in
a 3D model. Previously GeometryView was the one remaining corner-pair
output command not wired for `inf` (flagged as a trivial follow-up when
Snapshot/GeometryObjectsWrite were fixed earlier this session).
"""
from pathlib import Path

import h5py
import pytest

import gprMax

INF = float("inf")


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


def test_te_geometry_view_spans_full_invariant_thickness(tmp_path):
    scene = _scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    outfile = tmp_path / "te_view"
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, INF), p2=(0.02, 0.02, INF), dl=(1e-3, 1e-3, 1e-3), output_type="n", filename=str(outfile)
        )
    )
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run_te", hide_progress_bars=True)

    with h5py.File(str(outfile) + ".vtkhdf") as h:
        assert h["VTKHDF/CellData/Material"].shape == (2, 20, 20)


def test_tm_geometry_view_spans_full_invariant_thickness(tmp_path):
    scene = _scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    outfile = tmp_path / "tm_view"
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, INF), p2=(0.02, 0.02, INF), dl=(1e-3, 1e-3, 1e-3), output_type="n", filename=str(outfile)
        )
    )
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run_tm", hide_progress_bars=True)

    with h5py.File(str(outfile) + ".vtkhdf") as h:
        assert h["VTKHDF/CellData/Material"].shape == (1, 20, 20)


def test_3d_geometry_view_with_inf_is_rejected(tmp_path):
    scene = _scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, INF),
            p2=(0.02, 0.02, INF),
            dl=(1e-3, 1e-3, 1e-3),
            output_type="n",
            filename=str(tmp_path / "3d_view"),
        )
    )

    with pytest.raises(ValueError, match="2D"):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run_3d", hide_progress_bars=True)
