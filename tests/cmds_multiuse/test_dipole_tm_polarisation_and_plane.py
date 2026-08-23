"""Regression tests: in 2D TM mode, HertzianDipole and MagneticDipole
must be restricted to both the correct polarisation AND index 0 on the
invariant axis - not just polarisation.

- HertzianDipole (E-type source): polarisation must MATCH the invariant
  axis (Ez survives for TMz; Ex, Ey are forced pec). This restriction
  already existed; what's new is the plane-0 position check alongside it.
- MagneticDipole (H-type source): polarisation must be PERPENDICULAR to
  the invariant axis (Hx, Hy survive for TMz; Hz is never updated). The
  pre-existing check here was a straight copy of HertzianDipole's and
  was backwards - it rejected the only valid polarisations and allowed
  the dead one. Fixed alongside adding the same plane-0 check.

Confirmed directly from fields_updates_normal.pyx's 2D branches that
Hx/Hy (like Ez) are only ever computed at index 0 on the invariant axis
for TM - index 1 exists in the padded array but is dead, same as for E.
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
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


# --- HertzianDipole ------------------------------------------------------


def test_hertzian_dipole_correct_polarisation_and_plane_0_via_inf_is_accepted(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, INF), waveform_id="w"))
    _run(scene, tmp_path, "hd_ok")


def test_hertzian_dipole_correct_polarisation_wrong_plane_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.001), waveform_id="w"))
    with pytest.raises(ValueError, match="index 0"):
        _run(scene, tmp_path, "hd_bad_plane")


def test_hertzian_dipole_wrong_polarisation_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(0.01, 0.01, INF), waveform_id="w"))
    with pytest.raises(ValueError, match="polarisation"):
        _run(scene, tmp_path, "hd_bad_pol")


# --- MagneticDipole --------------------------------------------------------


def test_magnetic_dipole_correct_polarisation_and_plane_0_via_inf_is_accepted(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="x", p1=(0.01, 0.01, INF), waveform_id="w"))
    _run(scene, tmp_path, "md_ok_x")


def test_magnetic_dipole_other_perpendicular_polarisation_is_also_accepted(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="y", p1=(0.01, 0.01, INF), waveform_id="w"))
    _run(scene, tmp_path, "md_ok_y")


def test_magnetic_dipole_invariant_axis_polarisation_is_rejected(tmp_path):
    """The pre-existing bug: this used to be the ONLY polarisation
    accepted, when it should be the ONLY one rejected (Hz is dead for
    TMz - the tangential Hx/Hy are the survivors)."""
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="z", p1=(0.01, 0.01, INF), waveform_id="w"))
    with pytest.raises(ValueError, match="polarisation"):
        _run(scene, tmp_path, "md_bad_pol")


def test_magnetic_dipole_correct_polarisation_wrong_plane_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.MagneticDipole(polarisation="y", p1=(0.01, 0.01, 0.001), waveform_id="w"))
    with pytest.raises(ValueError, match="index 0"):
        _run(scene, tmp_path, "md_bad_plane")


# --- 3D unaffected -------------------------------------------------------


def test_3d_dipoles_unaffected_by_tm_guards(tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(0.01, 0.01, 0.015), waveform_id="w"))
    scene.add(gprMax.MagneticDipole(polarisation="z", p1=(0.01, 0.01, 0.005), waveform_id="w"))
    _run(scene, tmp_path, "dipoles_3d")
