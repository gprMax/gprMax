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

"""Regression tests for the vector plane-wave background guard.

A vector/angle plane wave assumes that the incident wave propagates in a
homogeneous background. Objects fully inside the TFSF box are allowed, but an
active TFSF correction node that uses incompatible electromagnetic properties
silently injects the incident field inconsistently.
"""

import logging

import pytest

import gprMax

INF = float("inf")


def _scene(mode):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.06, INF)))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="w"))
    scene.add(gprMax.Rx(p1=(0.03, 0.03, INF), outputs=["Ez" if mode == "TM" else "Ey"]))
    return scene


def _plane_wave(mode, material_id=None):
    kwargs = {
        "p1": (0.015, 0.015, INF),
        "p2": (0.045, 0.045, INF),
        "m_vec": (1, 1, 0),
        "psi": 90 if mode == "TM" else 0,
        "waveform_id": "w",
    }
    if material_id is not None:
        kwargs["material_id"] = material_id
    return gprMax.DiscretePlaneWaveVector(**kwargs)


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
        log_level=logging.WARNING,
    )


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_half_space_crossing_the_tfsf_boundary_is_rejected(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="halfspace"))
    scene.add(
        gprMax.Box(
            p1=(0.04, 0, INF),
            p2=(0.06, 0.06, INF),
            material_id="halfspace",
        )
    )
    scene.add(_plane_wave(mode))

    with pytest.raises(ValueError, match="homogeneous"):
        _run(scene, tmp_path, "halfspace")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_material_fully_inside_the_tfsf_box_is_allowed(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="block"))
    scene.add(
        gprMax.Box(
            p1=(0.025, 0.025, INF),
            p2=(0.035, 0.035, INF),
            material_id="block",
        )
    )
    scene.add(_plane_wave(mode))

    _run(scene, tmp_path, "inside")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_differently_named_free_space_background_is_allowed(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=1, se=0, mr=1, sm=0, id="vacuum"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, INF),
            p2=(0.06, 0.06, INF),
            material_id="vacuum",
        )
    )
    scene.add(_plane_wave(mode, material_id="vacuum"))

    _run(scene, tmp_path, "vacuum")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_material_at_the_lower_face_h_node_is_rejected(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=1, se=0, mr=4, sm=0, id="mr4"))
    scene.add(
        gprMax.Box(
            p1=(0.010, 0.010, INF),
            p2=(0.015, 0.050, INF),
            material_id="mr4",
        )
    )
    scene.add(_plane_wave(mode))

    with pytest.raises(ValueError, match="mr4|homogeneous"):
        _run(scene, tmp_path, "mr4")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_an_electric_edge_with_a_different_permeability_is_allowed(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=1, se=0, mr=4, sm=0, id="electric_edge"))
    if mode == "TM":
        p1, p2 = (0.045, 0.030, 0), (0.045, 0.030, 0.001)
    else:
        p1, p2 = (0.045, 0.030, 0), (0.045, 0.031, 0)
    scene.add(gprMax.Edge(p1=p1, p2=p2, material_id="electric_edge"))
    scene.add(_plane_wave(mode))

    _run(scene, tmp_path, "electric_edge")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_magnetic_edge_with_a_different_permittivity_is_allowed(tmp_path, mode):
    scene = _scene(mode)
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="magnetic_edge"))
    if mode == "TM":
        p1, p2 = (0.045, 0.030, 0), (0.045, 0.031, 0)
    else:
        p1, p2 = (0.045, 0.030, 0), (0.045, 0.030, 0.001)
    scene.add(gprMax.MagneticEdge(p1=p1, p2=p2, material_id="magnetic_edge"))
    scene.add(_plane_wave(mode))

    _run(scene, tmp_path, "magnetic_edge")


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_a_pec_object_on_the_tfsf_boundary_is_rejected(tmp_path, mode):
    scene = _scene(mode)
    scene.add(
        gprMax.Box(
            p1=(0.045, 0.015, INF),
            p2=(0.046, 0.045, INF),
            material_id="pec",
        )
    )
    scene.add(_plane_wave(mode))

    with pytest.raises(ValueError, match="pec|rigid|homogeneous"):
        _run(scene, tmp_path, "pec")


def test_a_pec_edge_on_the_tfsf_boundary_is_rejected(tmp_path):
    scene = _scene("TM")
    scene.add(
        gprMax.Edge(
            p1=(0.045, 0.030, 0),
            p2=(0.045, 0.030, 0.001),
            material_id="pec",
        )
    )
    scene.add(_plane_wave("TM"))

    with pytest.raises(ValueError, match="rigid|homogeneous"):
        _run(scene, tmp_path, "edge")
