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

"""Validation that a non-axial DPW TFSF box lies strictly inside the domain.

A box face that coincides with a domain boundary has no scattered-field
region beyond it, and the non-axial auxiliary 1D grid has no cells outside
the domain projection: the correction at that face writes onto the wall
plane (its H-side partner is structurally skipped there) while its 1D
sample index reads one cell outside the buffer - the kernels compile with
boundscheck(False), so that is a raw out-of-bounds read. Affected models
previously built "successfully" and produced silently wrong fields
(measured amplitude errors of 2.2x-33x). The non-axial commands now raise
at build time.

#plane_wave_axial is exempt: its auxiliary 1D grid is two-sided
(origin_axial = pml_length + buffercells of room beyond the domain
projection on both ends), so wall-spanning axial boxes are well-defined,
and they are the documented path for layered and half-space models.
"""

import pytest

import gprMax

INF = float("inf")

D = 1e-3
N = 30
L = N * D
PML = 8
A = (PML + 3) * D
B = (N - PML - 3) * D


def _scene_3d(wave, p1, p2):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(D, D, D)))
    scene.add(gprMax.Domain(p1=(L, L, L)))
    scene.add(gprMax.PMLThickness(thickness=PML))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=6e9, id="w"))
    scene.add(wave(p1, p2))
    return scene


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
        log_level=40,
    )


# --- angles command -------------------------------------------------------


@pytest.mark.parametrize("axis", range(3))
def test_angles_lower_wall_face_rejected(tmp_path, axis):
    lower = [A, A, A]
    lower[axis] = 0.0
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveAngles(
            p1=p1, p2=p2, theta=30, phi=40, psi=90, waveform_id="w"
        ),
        tuple(lower),
        (B, B, B),
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, f"angles_lo_{axis}")


@pytest.mark.parametrize("axis", range(3))
def test_angles_upper_wall_face_rejected(tmp_path, axis):
    upper = [B, B, B]
    upper[axis] = L
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveAngles(
            p1=p1, p2=p2, theta=30, phi=40, psi=90, waveform_id="w"
        ),
        (A, A, A),
        tuple(upper),
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, f"angles_hi_{axis}")


# --- vector command -------------------------------------------------------


def test_vector_wall_face_rejected(tmp_path):
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveVector(
            p1=p1, p2=p2, m_vec=(2, 1, 1), psi=30, waveform_id="w"
        ),
        (0.0, A, A),
        (B, B, B),
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "vector_lo_x")


# --- axial command is exempt ----------------------------------------------


def test_axial_box_spanning_domain_is_accepted(tmp_path):
    """The axial command's documented half-space pattern spans the domain:
    its two-sided 1D grid keeps valid sample indices beyond the faces."""
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveAxial(
            p1=p1, p2=p2, axis="z", psi=90, waveform_id="w"
        ),
        (A, A, 0.0),
        (B, B, L),
    )
    _run(scene, tmp_path, "axial_span_z")


# --- controls ---------------------------------------------------------------


def test_interior_box_is_accepted(tmp_path):
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveAngles(
            p1=p1, p2=p2, theta=30, phi=40, psi=90, waveform_id="w"
        ),
        (A, A, A),
        (B, B, B),
    )
    _run(scene, tmp_path, "angles_interior")


# --- 2D: free axes are validated, the invariant axis keeps its idiom -------


def test_2d_free_axis_wall_face_rejected(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(D, D, D)))
    scene.add(gprMax.Domain(p1=(L, L, INF)))
    scene.add(gprMax.PMLThickness(thickness=(PML, PML, 0, PML, PML, 0)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.0, A, INF), p2=(B, B, INF), theta=90, phi=0, psi=90, waveform_id="w"
        )
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "tm_x0")


# --- the legal boundary: faces exactly one cell inside are accepted --------


@pytest.mark.parametrize("side", ("lower", "upper"))
def test_box_face_one_cell_inside_the_boundary_is_accepted(tmp_path, side):
    """Pins the check's own boundary: face index 1 and n - 1 are legal, so
    the rule rejects only faces that coincide with the domain boundary."""
    p1 = [A, A, A]
    p2 = [B, B, B]
    if side == "lower":
        p1[0] = D
    else:
        p2[0] = L - D
    scene = _scene_3d(
        lambda p1, p2: gprMax.DiscretePlaneWaveAngles(
            p1=p1, p2=p2, theta=30, phi=40, psi=90, waveform_id="w"
        ),
        tuple(p1),
        tuple(p2),
    )
    _run(scene, tmp_path, f"edge_{side}")


pytestmark = pytest.mark.unit
