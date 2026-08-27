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

"""Construction and safety tests for portable one-axis RIPML slabs."""

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.hash_cmds_file import get_user_objects


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _base_scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.PMLThickness(thickness=0))
    return scene


def _add_pec_enclosure(scene, p1, p2, maximum_face):
    """Add four transverse walls and the maximum-stretch backing plate."""
    x0, y0, z0 = p1
    x1, y1, z1 = p2
    plates = (
        ((x0, y0, z0), (x1, y0, z1)),
        ((x0, y1, z0), (x1, y1, z1)),
        ((x0, y0, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x1, y1, z1)),
        (
            ((x0 if maximum_face == "x0" else x1), y0, z0),
            ((x0 if maximum_face == "x0" else x1), y1, z1),
        ),
    )
    for lower, upper in plates:
        scene.add(gprMax.Plate(p1=lower, p2=upper, material_id="pec"))


def test_boundary_replacement_uses_auto_id_and_global_profile(monkeypatch, tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.PMLSlab(
            p1=(0, 0, 0),
            p2=(0.01, 0.03, 0.03),
            maximum_face="x0",
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "boundary",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    pml = grid.pmls["slabs"][0]

    assert pml.ID == "internal_pml_1"
    assert pml.internal
    assert pml.formulation == "HORIPML"
    assert pml.ERA.shape == (1, 10)
    assert pml.HRA.shape == (1, 10)
    record = grid.pmls["internal_registry"][pml.ID]
    assert record["classification"] == "boundary-replacement"
    assert record["generated_pec_faces"] == 0
    assert record["enclosure_complete"]


def test_boundary_replacement_coefficients_match_native_pml(monkeypatch, tmp_path):
    captured = []
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured.append(self.G)

    monkeypatch.setattr(model_mod.Model, "build", patched_build)

    native = _base_scene()
    native.single_use_objects = [
        obj for obj in native.single_use_objects if not isinstance(obj, gprMax.PMLThickness)
    ]
    native.add(gprMax.PMLThickness(thickness=(5, 0, 0, 0, 0, 0)))
    gprMax.run(
        scenes=[native],
        geometry_only=True,
        outputfile=tmp_path / "native",
        hide_progress_bars=True,
    )

    replacement = _base_scene()
    replacement.add(
        gprMax.PMLSlab(
            p1=(0, 0, 0),
            p2=(0.005, 0.03, 0.03),
            maximum_face="x0",
        )
    )
    gprMax.run(
        scenes=[replacement],
        geometry_only=True,
        outputfile=tmp_path / "replacement",
        hide_progress_bars=True,
    )

    native_pml = captured[0].pmls["slabs"][0]
    replacement_pml = captured[1].pmls["slabs"][0]
    for name in ("ERA", "ERB", "ERE", "ERF", "HRA", "HRB", "HRE", "HRF"):
        np.testing.assert_array_equal(getattr(native_pml, name), getattr(replacement_pml, name))


def test_confined_internal_slab_retains_terminal_e_sample(monkeypatch, tmp_path):
    p1 = (0.005, 0.010, 0.010)
    p2 = (0.015, 0.020, 0.020)
    scene = _base_scene()
    scene.add(gprMax.PMLSlab(p1=p1, p2=p2, maximum_face="x0", id="feed_load"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "confined",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    pml = grid.pmls["slabs"][0]

    assert pml.EPhi1.shape[1:] == (11, 10, 11)
    assert pml.EPhi2.shape[1:] == (11, 11, 10)
    assert pml.ERA.shape == (1, 11)
    assert pml.HRA.shape == (1, 10)
    assert pml._electric_update_bounds() == (4, 15, 10, 20, 10, 20)
    record = grid.pmls["internal_registry"][pml.ID]
    assert record["classification"] == "internal-absorber"
    assert record["generated_pec_faces"] == 5
    assert record["enclosure_complete"]


@pytest.mark.parametrize("maximum_face", ("x0", "xmax", "y0", "ymax", "z0", "zmax"))
def test_automatic_pec_enclosure_supports_every_orientation(
    maximum_face, monkeypatch, tmp_path
):
    scene = _base_scene()
    scene.add(
        gprMax.PMLSlab(
            p1=(0.005, 0.008, 0.010),
            p2=(0.015, 0.020, 0.023),
            maximum_face=maximum_face,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / maximum_face,
        hide_progress_bars=True,
    )

    record = captured["grid"].pmls["internal_registry"]["internal_pml_1"]
    assert record["generated_pec_faces"] == 5
    assert record["enclosure_complete"]


def test_named_profile_is_local_to_slab(monkeypatch, tmp_path):
    scene = _base_scene()
    scene.add(gprMax.PMLFormulation(formulation="MRIPML", id="feed_profile"))
    scene.add(
        gprMax.PMLCFS(
            alphascalingprofile="constant",
            alphascalingdirection="forward",
            alphamin=0,
            alphamax=0,
            kappascalingprofile="constant",
            kappascalingdirection="forward",
            kappamin=1,
            kappamax=1,
            sigmascalingprofile="quadratic",
            sigmascalingdirection="forward",
            sigmamin=0,
            sigmamax=0.2,
            profile_id="feed_profile",
        )
    )
    scene.add(
        gprMax.PMLSlab(
            p1=(0, 0, 0),
            p2=(0.01, 0.03, 0.03),
            maximum_face="x0",
            profile_id="feed_profile",
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "profile",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    pml = grid.pmls["slabs"][0]

    assert grid.pmls["formulation"] == "HORIPML"
    assert pml.formulation == "MRIPML"
    assert pml.profile_id == "feed_profile"
    assert pml.CFS[0].sigma.scalingprofile == "quadratic"
    assert pml.CFS[0].sigma.max == 0.2


def test_exposed_transverse_faces_warn_when_automatic_pec_is_disabled(monkeypatch, tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.PMLSlab(
            p1=(0.005, 0.010, 0.010),
            p2=(0.015, 0.020, 0.020),
            maximum_face="x0",
            build_pec=False,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "invalid",
        hide_progress_bars=True,
    )

    record = captured["grid"].pmls["internal_registry"]["internal_pml_1"]
    assert not record["enclosure_complete"]
    assert record["generated_pec_faces"] == 0
    assert len(record["enclosure_warnings"]) == 5
    assert "exposed transverse" in record["enclosure_warnings"][0]
    assert "must have a PEC backing" in record["enclosure_warnings"][-1]


def test_manual_pec_enclosure_is_accepted_when_automatic_build_is_disabled(
    monkeypatch, tmp_path
):
    p1 = (0.005, 0.010, 0.010)
    p2 = (0.015, 0.020, 0.020)
    scene = _base_scene()
    _add_pec_enclosure(scene, p1, p2, "x0")
    scene.add(
        gprMax.PMLSlab(
            p1=p1,
            p2=p2,
            maximum_face="x0",
            build_pec=False,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "manual",
        hide_progress_bars=True,
    )

    record = captured["grid"].pmls["internal_registry"]["internal_pml_1"]
    assert record["generated_pec_faces"] == 0
    assert record["enclosure_complete"]
    assert record["enclosure_warnings"] == ()


def test_unknown_profile_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.PMLSlab(
            p1=(0, 0, 0),
            p2=(0.01, 0.03, 0.03),
            maximum_face="x0",
            profile_id="missing",
        )
    )
    with pytest.raises(ValueError, match="unknown profile"):
        gprMax.run(
            scenes=[scene],
            geometry_only=True,
            outputfile=tmp_path / "missing",
            hide_progress_bars=True,
        )


def test_hash_commands_accept_optional_profile_pec_flag_and_auto_slab_id():
    objects = get_user_objects(
        [
            "#pml_formulation: MRIPML feed\n",
            "#pml_cfs: constant forward 0 0 constant forward 1 1 quartic forward 0 None feed\n",
            "#pml_slab: 0 0 0 0.01 0.03 0.03 x0 feed n\n",
        ],
        checkessential=False,
    )
    formulations = [obj for obj in objects if isinstance(obj, gprMax.PMLFormulation)]
    slabs = [obj for obj in objects if isinstance(obj, gprMax.PMLSlab)]

    assert formulations[0].id == "feed"
    assert slabs[0].kwargs["profile_id"] == "feed"
    assert slabs[0].kwargs["build_pec"] is False


def test_hash_command_none_profile_allows_pec_flag():
    objects = get_user_objects(
        ["#pml_slab: 0 0 0 0.01 0.03 0.03 x0 None n\n"],
        checkessential=False,
    )
    slab = next(obj for obj in objects if isinstance(obj, gprMax.PMLSlab))

    assert slab.kwargs["profile_id"] is None
    assert slab.kwargs["build_pec"] is False
