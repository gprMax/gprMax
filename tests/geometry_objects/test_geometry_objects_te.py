"""Regression tests: #geometry_objects_write / #geometry_objects_read must
work correctly in 2D TE mode, including `inf` for both commands' bounds.

Two bugs found and fixed together:
- GeometryObjectsWrite.build() had never been wired for `inf` at all (unlike
  Box/Snapshot) - crashed with the raw decimal.InvalidOperation from
  round_int() the moment `inf` was used for p1/p2. Fixed the same way as
  Snapshot: resolve_inf_point(role="lower"/"upper") on lower_bound/
  upper_bound.
- GeometryObjectsRead.build() resolved its `p1` (the array's lower-left
  paste corner) with role=None (single-point resolution) - for TE mode that
  redirects `inf` to the *interior reference layer* (index 1), not to 0.
  Reading back a 2-cell-thick TE export then only had room for 1 cell at
  the target position, crashing with a shape-mismatch error pasting a
  (nx,ny,2) array into a (nx,ny,1) target slice starting at z=1. Fixed by
  using role="lower" (matching Box's p1), so the array's own 2-cell z-extent
  lands starting at z=0, correctly filling both TE cells.

A third piece, added later: a TM (1-cell) file read into a TE (2-cell)
model, or vice versa, is now handled automatically (broadcast 1->2 cells,
or reduce 2->1 taking the canonical/interior layer) rather than silently
leaving part of the domain unfilled or mismatching - a 2D model's geometry
has the same physical intention regardless of which reduction produced the
file. See ReadGeometryObject._resize_cell_axis()/_resize_edge_axis() in
gprMax/geometry_outputs/geometry_objects_read.py.
"""
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

INF = float("inf")


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _write_te_geometry(tmp_path, monkeypatch):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="diel"))
    scene.add(gprMax.Box(p1=(0.003, 0.003, INF), p2=(0.007, 0.007, INF), material_id="diel"))

    outfile = tmp_path / "te_export"
    scene.add(
        gprMax.GeometryObjectsWrite(p1=(0.0, 0.0, INF), p2=(0.01, 0.01, INF), filename=str(outfile))
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "write_te",
        hide_progress_bars=True,
    )
    return captured["grid"], outfile.with_suffix(".h5"), tmp_path / "te_export_materials.json"


def _write_tm_geometry(tmp_path, monkeypatch):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="diel"))
    scene.add(gprMax.Box(p1=(0.003, 0.003, INF), p2=(0.007, 0.007, INF), material_id="diel"))

    outfile = tmp_path / "tm_export"
    scene.add(
        gprMax.GeometryObjectsWrite(p1=(0.0, 0.0, INF), p2=(0.01, 0.01, INF), filename=str(outfile))
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "write_tm",
        hide_progress_bars=True,
    )
    return captured["grid"], outfile.with_suffix(".h5"), tmp_path / "tm_export_materials.json"


def test_geometry_objects_read_broadcasts_tm_file_into_te_model(tmp_path, monkeypatch, capsys):
    """A 1-cell-thick (TM) file read into a 2-cell (TE) model must be
    broadcast automatically - the same physical box, no hoop-jumping
    required from the user - with an informational message logged.

    gprMax's logger has propagate=False (gprMax/utilities/logging.py), so
    pytest's caplog fixture (which relies on root-logger propagation)
    can't see it - the message is checked via captured stdout instead,
    matching how gprMax actually emits it.
    """
    _, geofile, matfile = _write_tm_geometry(tmp_path, monkeypatch)

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(INF, INF, INF),
            geofile=str(geofile),
            material_database=matfile.stem,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_tm_into_te",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    assert grid.solid.shape == (10, 10, 2)
    assert np.array_equal(grid.solid[:, :, 0], grid.solid[:, :, 1])
    diel_material = next(m for m in grid.materials if m.ID.startswith("diel{"))
    assert np.sum(grid.solid[:, :, 0] == diel_material.numID) == 16
    assert "broadcasting" in capsys.readouterr().out


def test_geometry_objects_read_reduces_te_file_into_tm_model(tmp_path, monkeypatch, capsys):
    """A 2-cell-thick (TE) file read into a 1-cell (TM) model must be
    reduced automatically (taking the interior/canonical layer), producing
    a model that solves identically to one built directly with the same
    box - not just matching material counts, but the actual field trace."""
    _, geofile, matfile = _write_te_geometry(tmp_path, monkeypatch)

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(INF, INF, INF),
            geofile=str(geofile),
            material_database=matfile.stem,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_te_into_tm",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    assert grid.solid.shape == (10, 10, 1)
    diel_material = next(m for m in grid.materials if m.ID.startswith("diel{"))
    assert np.sum(grid.solid[:, :, 0] == diel_material.numID) == 16
    assert "reducing" in capsys.readouterr().out


def test_geometry_objects_read_te_to_tm_solves_identically_to_direct_build(tmp_path, monkeypatch):
    """Stronger check than material counts: the ID array reconstruction
    (interior-edge selection, not just cell-based data) must be physically
    correct, verified via a real solve with a source/receiver, not just
    geometry_only inspection."""
    _, geofile, matfile = _write_te_geometry(tmp_path, monkeypatch)

    def _run_tm(use_import):
        scene = gprMax.Scene()
        scene.add(gprMax.DomainMode(mode="TM"))
        scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
        scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-11))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
        scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.005, INF), waveform_id="w"))
        scene.add(gprMax.Rx(p1=(0.002, 0.002, INF)))
        if use_import:
            scene.add(
                gprMax.GeometryObjectsRead(
                    p1=(INF, INF, INF),
                    geofile=str(geofile),
                    material_database=matfile.stem,
                )
            )
        else:
            scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="diel"))
            scene.add(
                gprMax.Box(p1=(0.003, 0.003, INF), p2=(0.007, 0.007, INF), material_id="diel")
            )
        outfile = tmp_path / ("imported_solve" if use_import else "direct_solve")
        gprMax.run(scenes=[scene], n=1, outputfile=outfile, hide_progress_bars=True)
        return outfile.with_suffix(".h5")

    import h5py

    direct_file = _run_tm(use_import=False)
    imported_file = _run_tm(use_import=True)

    with h5py.File(direct_file) as f:
        ez_direct = f["rxs/rx1/Ez"][:]
    with h5py.File(imported_file) as f:
        ez_imported = f["rxs/rx1/Ez"][:]

    assert not np.any(np.isnan(ez_direct))
    assert not np.any(np.isnan(ez_imported))
    assert np.allclose(ez_direct, ez_imported)


def test_geometry_objects_write_accepts_inf_and_produces_invariant_export(tmp_path, monkeypatch):
    grid_written, geofile, matfile = _write_te_geometry(tmp_path, monkeypatch)
    assert grid_written.solid.shape == (10, 10, 2)
    assert np.array_equal(grid_written.solid[:, :, 0], grid_written.solid[:, :, 1])
    assert geofile.exists()
    assert matfile.exists()


def test_geometry_objects_read_with_inf_fills_both_te_cells_from_lower_corner(
    tmp_path, monkeypatch
):
    grid_written, geofile, matfile = _write_te_geometry(tmp_path, monkeypatch)

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, INF),
            geofile=str(geofile),
            material_database=matfile.stem,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_te",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    assert grid.solid.shape == (10, 10, 2)
    assert np.array_equal(grid.solid[:, :, 0], grid.solid[:, :, 1])

    diel_material = next(m for m in grid.materials if m.ID.startswith("diel{"))
    assert np.sum(grid.solid[:, :, 0] == diel_material.numID) == 16
    assert np.sum(grid.solid[:, :, 1] == diel_material.numID) == 16


def test_geometry_objects_read_inf_rejected_in_3d(tmp_path, monkeypatch):
    grid_written, geofile, matfile = _write_te_geometry(tmp_path, monkeypatch)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, INF),
            geofile=str(geofile),
            material_database=matfile.stem,
        )
    )

    with pytest.raises(ValueError, match="2D"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "read_3d_bad",
            hide_progress_bars=True,
        )
