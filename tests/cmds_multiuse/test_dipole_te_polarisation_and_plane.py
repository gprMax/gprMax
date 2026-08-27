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

"""Regression tests: in 2D TE mode, HertzianDipole and MagneticDipole
must be restricted to both the correct polarisation AND index 1 (the
interior layer) on the invariant axis - the exact E<->H dual of the TM
restrictions in tests/cmds_multiuse/test_dipole_tm_polarisation_and_plane.py.

- HertzianDipole (E-type source): polarisation must be PERPENDICULAR to
  the invariant axis (Ex, Ey survive for TEz; Ez is forced pec).
- MagneticDipole (H-type source): polarisation must MATCH the invariant
  axis (Hz survives for TEz; Hx, Hy are forced pmc).
- Both: the surviving component is only ever computed at the interior
  index 1 on the invariant axis - index 0 and 2 are the outer walls,
  forced pec/pmc by tex()/tey()/tez() (confirmed directly from that
  code, including its defensive forcing of the survivor components at
  the outer walls specifically so this restriction holds).
"""
import tempfile
from pathlib import Path

import pytest

import gprMax

INF = float("inf")


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


def _base_scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


# --- HertzianDipole ------------------------------------------------------


@pytest.mark.parametrize("polarisation", ["x", "y"])
def test_hertzian_dipole_perpendicular_polarisation_and_plane_1_via_inf_is_accepted(
    tmp_path, polarisation
):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation=polarisation, p1=(0.01, 0.01, INF), waveform_id="w"))
    _run(scene, tmp_path, f"hd_ok_{polarisation}")


def test_hertzian_dipole_invariant_axis_polarisation_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, INF), waveform_id="w"))
    with pytest.raises(ValueError, match="polarisation"):
        _run(scene, tmp_path, "hd_bad_pol")


def test_hertzian_dipole_correct_polarisation_wrong_plane_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(0.01, 0.01, 0.0), waveform_id="w"))
    with pytest.raises(ValueError, match="index 1"):
        _run(scene, tmp_path, "hd_bad_plane")


# --- MagneticDipole --------------------------------------------------------


def test_magnetic_dipole_invariant_polarisation_and_plane_1_via_inf_is_accepted(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="z", p1=(0.01, 0.01, INF), waveform_id="w"))
    _run(scene, tmp_path, "md_ok")


@pytest.mark.parametrize("polarisation", ["x", "y"])
def test_magnetic_dipole_perpendicular_polarisation_is_rejected(tmp_path, polarisation):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation=polarisation, p1=(0.01, 0.01, INF), waveform_id="w"))
    with pytest.raises(ValueError, match="polarisation"):
        _run(scene, tmp_path, f"md_bad_pol_{polarisation}")


def test_magnetic_dipole_correct_polarisation_wrong_plane_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="z", p1=(0.01, 0.01, 0.002), waveform_id="w"))
    with pytest.raises(ValueError, match="index 1"):
        _run(scene, tmp_path, "md_bad_plane")
