"""End-to-end tests for `inf` coordinates in #snapshot / Snapshot,
covering the same range-endpoint resolution rules already proven for
Box (gprMax/user_inputs.py's resolve_inf_point(), role="lower"/"upper"):
purely positional, spans the full invariant-axis thickness in TM/TE 2D
mode with no special-casing, and rejected outright in a 3D model.
"""
import glob
import tempfile
from pathlib import Path

import h5py
import pytest

import gprMax

INF = float("inf")


def _run(scene, tmp_path, label):
    # Snapshots fire during time-stepping, not at geometry-build time, so
    # this needs a real solve (not geometry_only=True).
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    files = sorted(glob.glob(str(tmp_path / f"{label}_snaps" / "*")))
    return files


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def test_te_snapshot_uses_only_live_invariant_plane(tmp_path):
    scene = _scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, INF), p2=(0.02, 0.02, INF), dl=(1e-3, 1e-3, 1e-3), filename="snap", time=5e-13
        )
    )

    files = _run(scene, tmp_path, "te_snap")
    assert len(files) == 1
    with h5py.File(files[0]) as h:
        assert h["VTKHDF/CellData/Ey"].shape == (1, 20, 20)
        # VTK stores cell data, so a z-origin of dl/2 places the centre of
        # this single output cell on the live TE plane at z=dl (index 1).
        assert h["VTKHDF"].attrs["Origin"][2] == pytest.approx(0.5e-3)


def test_tm_snapshot_spans_full_invariant_thickness(tmp_path):
    scene = _scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, INF), p2=(0.02, 0.02, INF), dl=(1e-3, 1e-3, 1e-3), filename="snap", time=5e-13
        )
    )

    files = _run(scene, tmp_path, "tm_snap")
    assert len(files) == 1
    with h5py.File(files[0]) as h:
        assert h["VTKHDF/CellData/Ez"].shape == (1, 20, 20)


def test_3d_snapshot_with_inf_is_rejected(tmp_path):
    scene = _scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, INF), p2=(0.02, 0.02, INF), dl=(1e-3, 1e-3, 1e-3), filename="snap", time=5e-13
        )
    )

    with pytest.raises(ValueError, match="2D"):
        _run(scene, tmp_path, "3d_snap")


def test_te_snapshot_rejects_bounds_that_exclude_live_plane(tmp_path):
    scene = _scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, 0),
            p2=(0.02, 0.02, 1e-3),
            dl=(1e-3, 1e-3, 1e-3),
            filename="snap",
            time=5e-13,
        )
    )

    with pytest.raises(ValueError, match="live z-index 1"):
        _run(scene, tmp_path, "te_dead_plane")
