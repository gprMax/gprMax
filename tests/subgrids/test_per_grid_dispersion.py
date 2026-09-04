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

"""Integration coverage for grid-local dispersive storage and updates."""

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_module


DL = 0.003
DOMAIN = (0.06, 0.06, 0.06)
SUBGRID_P1 = (0.021, 0.021, 0.021)
SUBGRID_P2 = (0.039, 0.039, 0.039)


def _add_debye_material(owner, material_id, poles=3):
    owner.add(gprMax.Material(er=3.5, se=0.005, mr=1, sm=0, id=material_id))
    owner.add(
        gprMax.AddDebyeDispersion(
            poles=poles,
            er_delta=(2.0, 1.0, 0.5)[:poles],
            tau=(1e-9, 2e-10, 5e-11)[:poles],
            material_ids=(material_id,),
        )
    )


def _base_scene(iterations=5):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(n=1))
    subgrid = gprMax.SubGridHSG(
        p1=SUBGRID_P1,
        p2=SUBGRID_P2,
        ratio=1,
        id="local_grid",
    )
    scene.add(subgrid)
    return scene, subgrid


def _capture_grids(monkeypatch):
    captured = {}
    original_build = model_module.Model.build

    def patched_build(self):
        original_build(self)
        captured["main"] = self.G
        captured["subgrid"] = self.subgrids[0]

    monkeypatch.setattr(model_module.Model, "build", patched_build)
    return captured


@pytest.mark.integration
@pytest.mark.parametrize(
    "placement,expected_poles",
    [
        ("subgrid", (0, 3)),
        ("main", (3, 0)),
        ("crossing", (2, 2)),
    ],
)
def test_dispersive_arrays_are_allocated_only_on_grids_that_need_them(
    monkeypatch, tmp_path, placement, expected_poles
):
    scene, subgrid = _base_scene()

    if placement == "subgrid":
        _add_debye_material(subgrid, "target")
        subgrid.add(gprMax.Sphere(p1=(0.03, 0.03, 0.03), r=0.006, material_id="target"))
    elif placement == "main":
        _add_debye_material(scene, "target")
        scene.add(gprMax.Sphere(p1=(0.012, 0.03, 0.03), r=0.006, material_id="target"))
    else:
        # A continuous half-space is represented on each side of the HSG
        # interface by geometry owned by that grid. Both grids must therefore
        # retain their independent pole histories.
        _add_debye_material(scene, "main_half_space", poles=2)
        scene.add(
            gprMax.Box(
                p1=(0.03, 0, 0),
                p2=DOMAIN,
                material_id="main_half_space",
            )
        )
        _add_debye_material(subgrid, "subgrid_half_space", poles=2)
        subgrid.add(
            gprMax.Box(
                p1=(0.03, SUBGRID_P1[1], SUBGRID_P1[2]),
                p2=SUBGRID_P2,
                material_id="subgrid_half_space",
            )
        )

    captured = _capture_grids(monkeypatch)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / placement,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    for grid, poles in zip((captured["main"], captured["subgrid"]), expected_poles):
        assert grid.maxpoles == poles
        assert hasattr(grid, "Tx") is (poles > 0)
        assert hasattr(grid, "updatecoeffsdispersive") is (poles > 0)
        if poles:
            assert grid.Tx.shape[0] == poles
            assert grid.updatecoeffsdispersive.shape[1] == 3 * poles
            assert grid.mem_est_dispersive() == grid.Tx.nbytes * 3
        else:
            assert grid.mem_est_dispersive() == 0


def _parity_scene(*, use_subgrid):
    dl = 0.002
    domain = (0.12, 0.12, 0.12)
    subgrid_p1 = (0.038, 0.038, 0.038)
    subgrid_p2 = (0.082, 0.082, 0.082)
    centre = (0.06, 0.06, 0.06)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(iterations=300))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.DispersiveAveraging(enabled=True))

    owner = scene
    if use_subgrid:
        owner = gprMax.SubGridHSG(
            p1=subgrid_p1,
            p2=subgrid_p2,
            ratio=1,
            id="local_grid",
        )
        scene.add(owner)

    _add_debye_material(owner, "target")
    owner.add(gprMax.Sphere(p1=centre, r=0.014, material_id="target"))
    owner.add(
        gprMax.Rx(
            p1=centre,
            id="inside",
            outputs=["Ez", "Hy", "Iz"],
        )
    )
    owner.add(
        gprMax.Snapshot(
            p1=(0.056, 0.056, 0.056),
            p2=(0.064, 0.064, 0.064),
            dl=(dl, dl, dl),
            filename="parity_fields",
            iterations=160,
            fileext=".h5",
            outputs=["Ez", "Hy"],
        )
    )
    scene.add(gprMax.Waveform(wave_type="gaussianprime", amp=1, freq=1.5e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=(0.024, 0.06, 0.06),
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.096, 0.06, 0.06), id="rx"))
    return scene


def _receiver(group, name):
    """Return a receiver group by its public ID rather than list position."""

    return next(rx for rx in group["rxs"].values() if rx.attrs.get("Name") == name)


@pytest.mark.integration
@pytest.mark.parametrize(
    "precision,relative_tolerance",
    [("single", 5e-4), ("double", 1e-10)],
)
def test_localised_debye_subgrid_matches_uniform_grid_trace(
    tmp_path, precision, relative_tolerance
):
    plain_output = tmp_path / "plain"
    subgrid_output = tmp_path / "subgrid"

    gprMax.run(
        scenes=[_parity_scene(use_subgrid=False)],
        outputfile=plain_output,
        cpu_precision=precision,
        hide_progress_bars=True,
    )
    gprMax.run(
        scenes=[_parity_scene(use_subgrid=True)],
        outputfile=subgrid_output,
        subgrid=True,
        autotranslate=True,
        cpu_precision=precision,
        hide_progress_bars=True,
    )

    with h5py.File(plain_output.with_suffix(".h5"), "r") as plain, h5py.File(
        subgrid_output.with_suffix(".h5"), "r"
    ) as localised:
        plain_external = _receiver(plain, "rx")
        subgrid_external = _receiver(localised, "rx")
        plain_internal = _receiver(plain, "inside")
        embedded = localised["subgrids/local_grid"]
        subgrid_internal = _receiver(embedded, "inside")

        assert embedded.attrs["coupling_mode"] == "equal_resolution"
        assert embedded.attrs["subgrid_pml_thickness"] == 0
        assert embedded.attrs["interpolation"] == 0
        assert not bool(embedded.attrs["filter"])
        expected_dtype = np.dtype(np.float32 if precision == "single" else np.float64)
        assert subgrid_internal["Ez"].dtype == expected_dtype

        for component in ("Ez", "Hy"):
            reference = plain_external[component][:]
            actual = subgrid_external[component][:]
            assert np.max(np.abs(reference)) > 0
            relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            assert relative_l2 < relative_tolerance

        # Internal H and the Ampere-loop current derived from H used to be
        # written one fine-grid sample early.  At ratio 1 the corrected
        # histories must agree directly, without any manual trace shift.
        for component in ("Ez", "Hy", "Iz"):
            reference = plain_internal[component][:]
            actual = subgrid_internal[component][:]
            assert np.max(np.abs(reference)) > 0
            relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            assert relative_l2 < relative_tolerance

        dt = float(localised["subgrids/local_grid"].attrs["dt"])
        assert subgrid_internal["Ez"].attrs["TimeSampleOffset"] == 0.0
        assert subgrid_internal["Hy"].attrs["TimeSampleOffset"] == pytest.approx(-0.5 * dt)
        assert subgrid_internal["Iz"].attrs["TimeSampleOffset"] == pytest.approx(-0.5 * dt)

    plain_snapshot = tmp_path / "plain_snaps" / "parity_fields.h5"
    subgrid_snapshot = tmp_path / "subgrid_snaps" / "parity_fields.h5"
    with h5py.File(plain_snapshot, "r") as reference, h5py.File(
        subgrid_snapshot, "r"
    ) as actual:
        assert actual.attrs["iteration"] == reference.attrs["iteration"] == 160
        assert actual.attrs["magnetic_time"] == pytest.approx(
            reference.attrs["magnetic_time"]
        )
        for component in ("Ez", "Hy"):
            expected = reference[component][:]
            observed = actual[component][:]
            assert np.max(np.abs(expected)) > 0
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < relative_tolerance
