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

"""Field-output and internal-PML coverage for objects owned by an HSG subgrid."""

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod


def _subgrid_scene(timewindow=5e-11):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=timewindow))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    return scene, subgrid


@pytest.mark.integration
def test_snapshot_runs_at_fine_time_step_and_uses_global_origin(tmp_path):
    scene, subgrid = _subgrid_scene()
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    subgrid.add(
        gprMax.HertzianDipole(
            p1=(0.045, 0.045, 0.045),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    subgrid.add(
        gprMax.Snapshot(
            p1=(0.042, 0.042, 0.042),
            p2=(0.049, 0.049, 0.049),
            dl=(0.001, 0.001, 0.001),
            filename="subgrid_fields",
            time=2e-11,
            fileext=".h5",
            outputs=["Ez"],
        )
    )
    subgrid.add(
        gprMax.Snapshot(
            p1=(0.042, 0.042, 0.042),
            p2=(0.049, 0.049, 0.049),
            dl=(0.001, 0.001, 0.001),
            filename="subgrid_fields_initial",
            iterations=0,
            fileext=".h5",
            outputs=["Ez", "Hy"],
        )
    )
    subgrid.add(
        gprMax.Snapshot(
            p1=(0.042, 0.042, 0.042),
            p2=(0.049, 0.049, 0.049),
            dl=(0.001, 0.001, 0.001),
            filename="subgrid_fields_iteration",
            iterations=10,
            fileext=".h5",
            outputs=["Ez"],
        )
    )

    output = tmp_path / "snapshot_model"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    filename = tmp_path / "snapshot_model_snaps" / "subgrid_fields.h5"
    iteration_filename = (
        tmp_path / "snapshot_model_snaps" / "subgrid_fields_iteration.h5"
    )
    initial_filename = tmp_path / "snapshot_model_snaps" / "subgrid_fields_initial.h5"
    assert filename.exists()
    assert iteration_filename.exists()
    assert initial_filename.exists()
    with h5py.File(filename, "r") as snapshot, h5py.File(
        iteration_filename, "r"
    ) as iteration_snapshot, h5py.File(initial_filename, "r") as initial_snapshot:
        np.testing.assert_allclose(snapshot.attrs["origin"], (0.042, 0.042, 0.042))
        np.testing.assert_allclose(snapshot.attrs["dx_dy_dz"], (0.001, 0.001, 0.001))
        assert snapshot.attrs["time"] == pytest.approx(2e-11, rel=0.1)
        assert snapshot.attrs["iteration"] == 10
        assert snapshot.attrs["magnetic_time"] == pytest.approx(0.95 * snapshot.attrs["time"])
        assert snapshot["Ez"].shape == (7, 7, 7)
        assert np.max(np.abs(snapshot["Ez"][...])) > 0
        assert iteration_snapshot.attrs["iteration"] == snapshot.attrs["iteration"]
        assert iteration_snapshot.attrs["time"] == snapshot.attrs["time"]
        np.testing.assert_array_equal(iteration_snapshot["Ez"][...], snapshot["Ez"][...])
        assert initial_snapshot.attrs["iteration"] == 0
        assert initial_snapshot.attrs["time"] == 0.0
        assert initial_snapshot.attrs["magnetic_time"] == pytest.approx(
            -0.5 * snapshot.attrs["time"] / snapshot.attrs["iteration"]
        )
        assert not np.any(initial_snapshot["Ez"][...])
        assert not np.any(initial_snapshot["Hy"][...])


def _capture_grids(monkeypatch):
    captured = []
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured.append([self.G, *self.subgrids])

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _add_internal_slab(owner):
    owner.add(
        gprMax.PMLSlab(
            p1=(0.040, 0.040, 0.040),
            p2=(0.044, 0.050, 0.050),
            maximum_face="x0",
            id="fine_load",
        )
    )


@pytest.mark.integration
def test_subgrid_internal_pml_matches_uniform_fine_grid_coefficients(monkeypatch, tmp_path):
    captured = _capture_grids(monkeypatch)

    uniform = gprMax.Scene()
    uniform.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    uniform.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    uniform.add(gprMax.TimeWindow(time=1e-11))
    uniform.add(gprMax.PMLThickness(thickness=0))
    _add_internal_slab(uniform)
    gprMax.run(
        scenes=[uniform],
        geometry_only=True,
        outputfile=tmp_path / "uniform",
        hide_progress_bars=True,
        cpu_precision="double",
    )

    subgrid_scene, subgrid = _subgrid_scene(timewindow=1e-11)
    _add_internal_slab(subgrid)
    gprMax.run(
        scenes=[subgrid_scene],
        geometry_only=True,
        outputfile=tmp_path / "subgrid",
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    uniform_pml = next(pml for pml in captured[0][0].pmls["slabs"] if pml.ID == "fine_load")
    fine_grid = captured[1][1]
    subgrid_pml = next(pml for pml in fine_grid.pmls["slabs"] if pml.ID == "fine_load")
    for name in ("ERA", "ERB", "ERE", "ERF", "HRA", "HRB", "HRE", "HRF"):
        np.testing.assert_allclose(getattr(subgrid_pml, name), getattr(uniform_pml, name))

    record = fine_grid.pmls["internal_registry"]["fine_load"]
    assert record["generated_pec_faces"] == 5
    assert record["enclosure_complete"]


def test_subgrid_internal_pml_cannot_overlap_coupling_region(tmp_path):
    scene, subgrid = _subgrid_scene(timewindow=1e-11)
    subgrid.add(
        gprMax.PMLSlab(
            p1=(0.029, 0.040, 0.040),
            p2=(0.034, 0.050, 0.050),
            maximum_face="x0",
        )
    )

    with pytest.raises(ValueError, match="wholly inside the subgrid working region"):
        gprMax.run(
            scenes=[scene],
            geometry_only=True,
            outputfile=tmp_path / "invalid",
            subgrid=True,
            autotranslate=True,
            hide_progress_bars=True,
        )


@pytest.mark.integration
def test_subgrid_internal_pml_runs_on_every_fine_update(tmp_path):
    scene, subgrid = _subgrid_scene(timewindow=1e-10)
    _add_internal_slab(subgrid)
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    subgrid.add(
        gprMax.HertzianDipole(
            p1=(0.047, 0.045, 0.045),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    subgrid.add(gprMax.Rx(p1=(0.043, 0.045, 0.045), id="inside_load"))

    output = tmp_path / "pml_update"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as result:
        ez = result["subgrids/fine_grid/rxs/rx1/Ez"][...]
        assert np.all(np.isfinite(ez))
        assert np.max(np.abs(ez)) > 0
