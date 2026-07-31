"""Tests for the Discrete Plane Wave (DPW) sources in 2D TM/TE modes -
builder validation, the vector-mode angle-convention regression, and
solver-level physics (dead components stay dead; the 4-edge 2D TFSF closes).

Key design points under test (see gprMax/cython/plane_wave.pyx and
DiscretePlaneWave in gprMax/sources.py):

- A 2D TFSF is a rectangle of 4 edges, not a box of 6 faces: the face pair
  normal to the invariant axis is skipped via the `skip_axis` parameter.
  Without the skip, spurious corrections are written into structurally-dead
  field components (and, for TM corners at wall index 0, out of bounds).
- In-plane propagation is a hard requirement (m[invariant] == 0 after the
  integer mapping) - also a stability requirement under the larger 2D CFL
  timestep.
- Polarisation is validated operationally on the projections (the mode's
  dead-component incident projections must be exactly 0) rather than via a
  closed-form psi rule.
- The vector-mode angle computation follows the wavefront-normal convention
  phys = m / [dx,dy,dz] (matching find_dpw_integers_optimized) - an earlier
  version had theta/phi slots swapped AND the m*d (cell-diagonal) convention,
  both silently wrong in 3D; pinned here including an anisotropic-cell case.
- Exact in-plane angles (theta = 90) must not overflow the integer mapper
  (cos(90 deg) evaluates to ~6e-17, not 0; snapped before the search).
"""
import math

import h5py
import numpy as np
import pytest

import gprMax

INF = float("inf")


def _scene_2d(mode="TM"):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.06, INF)))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="w"))
    return scene


def _angles_dpw(theta, phi, psi):
    return gprMax.DiscretePlaneWaveAngles(
        p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
        theta=theta, phi=phi, psi=psi, waveform_id="w",
    )


def _run(scene, tmp_path, label, geometry_only=True):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=geometry_only,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


# --- Builder validation: angles mode -------------------------------------


def test_tmz_in_plane_angles_accepted_with_inf_corners(tmp_path):
    scene = _scene_2d("TM")
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=90))
    _run(scene, tmp_path, "tmz_ok")


def test_tmz_out_of_plane_theta_rejected(tmp_path):
    scene = _scene_2d("TM")
    scene.add(_angles_dpw(theta=60, phi=26.565051177, psi=90))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "tmz_theta")


def test_tmz_wrong_psi_rejected_naming_dead_components(tmp_path):
    # psi=0 gives in-plane E - dead in TM.
    scene = _scene_2d("TM")
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=0))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "tmz_psi")


def test_tez_in_plane_angles_accepted(tmp_path):
    scene = _scene_2d("TE")
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=0))
    _run(scene, tmp_path, "tez_ok")


def test_tez_wrong_psi_rejected(tmp_path):
    # psi=90 gives E along z - dead in TE.
    scene = _scene_2d("TE")
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=90))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "tez_psi")


# --- Builder validation: vector mode --------------------------------------


def test_tmz_in_plane_vector_accepted(tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
            m_vec=(1, 2, 0), psi=90, waveform_id="w",
        )
    )
    _run(scene, tmp_path, "vec_ok")


def test_tmz_out_of_plane_vector_rejected(tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
            m_vec=(1, 2, 3), psi=90, waveform_id="w",
        )
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "vec_bad")


# --- Builder validation: axial mode ---------------------------------------


def test_axial_invariant_axis_rejected(tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
            axis="z", psi=90, waveform_id="w",
        )
    )
    # gprMax raises bare ValueError (the message goes to its logger, which
    # has propagate=False) - so no match= is possible here or below.
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "axial_bad")


def test_axial_in_plane_axis_accepted(tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
            axis="x", psi=90, waveform_id="w",
        )
    )
    _run(scene, tmp_path, "axial_ok")


# --- Corner handling ------------------------------------------------------


def test_explicit_invariant_corner_overridden_with_warning(tmp_path, capsys):
    """Explicitly-typed invariant-axis coordinates are accepted but
    overridden to the mode-determined extent, with a visible warning
    teaching the `inf` idiom. gprMax's logger has propagate=False so the
    warning is checked via captured stdout, matching how it is emitted."""
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.015, 0.015, 0.0), p2=(0.045, 0.045, 0.0),
            theta=90, phi=26.565051177, psi=90, waveform_id="w",
        )
    )
    _run(scene, tmp_path, "corner_warn")
    out = capsys.readouterr().out
    assert "overridden" in out and "invariant" in out


def test_inverted_in_plane_corners_rejected(tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.045, 0.015, INF), p2=(0.015, 0.045, INF),
            theta=90, phi=26.565051177, psi=90, waveform_id="w",
        )
    )
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "corner_inverted")


# --- 3D vector-mode angle-convention regression ---------------------------


def _build_3d_vector_dpw(tmp_path, m_vec, dl):
    """Builds a 3D scene with a vector-mode DPW and returns the built
    source object for direct inspection."""
    import gprMax.model as model_mod

    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    model_mod.Model.build = patched_build
    try:
        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=dl))
        scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-11))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
        scene.add(
            gprMax.DiscretePlaneWaveVector(
                p1=(0.007, 0.007, 0.007), p2=(0.021, 0.021, 0.021),
                m_vec=m_vec, psi=90, waveform_id="w",
            )
        )
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True,
            outputfile=tmp_path / "vec3d", hide_progress_bars=True,
        )
        return captured["grid"].discreteplanewaves[0]
    finally:
        model_mod.Model.build = orig_build


def test_3d_vector_cubic_cells_angles_and_ds(tmp_path):
    """m=(1,1,1), cubic cells: theta = acos(1/sqrt(3)) = 54.7356 deg,
    phi = 45 deg, ds = dx/sqrt(3). Pins the slot-swap fix (an earlier
    version reported these with theta and phi exchanged)."""
    dpw = _build_3d_vector_dpw(tmp_path, (1, 1, 1), dl=(1e-3, 1e-3, 1e-3))
    assert dpw.actual_angles[0] == pytest.approx(math.degrees(math.acos(1 / math.sqrt(3))))
    assert dpw.actual_angles[1] == pytest.approx(45.0)
    assert dpw.ds == pytest.approx(1e-3 / math.sqrt(3))


def test_3d_vector_anisotropic_cells_wavefront_normal_convention(tmp_path):
    """m=(1,0,1) with dz=2*dx: the physical propagation direction is the
    wavefront normal (m_x/dx, m_y/dy, m_z/dz) -> theta = 63.435 deg (NOT
    the 26.565 deg the old cell-diagonal m*d convention produced). Pins
    the anisotropic-cell theta fix against the authoritative convention
    used by find_dpw_integers_optimized."""
    dpw = _build_3d_vector_dpw(tmp_path, (1, 0, 1), dl=(1e-3, 1e-3, 2e-3))
    assert dpw.actual_angles[0] == pytest.approx(63.43494882, abs=1e-6)
    assert dpw.actual_angles[1] == pytest.approx(0.0, abs=1e-9)


def test_3d_in_plane_theta_90_does_not_overflow_integer_mapper(tmp_path):
    """Regression: theta=90 exactly (cos -> ~6e-17) previously made the
    continued-fraction search chase the residue with astronomically large
    integers until integer conversion overflowed - in 3D too."""
    import gprMax.model as model_mod

    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    model_mod.Model.build = patched_build
    try:
        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
        scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
        scene.add(gprMax.TimeWindow(time=1e-11))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
        scene.add(
            gprMax.DiscretePlaneWaveAngles(
                p1=(0.007, 0.007, 0.007), p2=(0.021, 0.021, 0.021),
                theta=90, phi=26.565051177, psi=90, waveform_id="w",
            )
        )
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True,
            outputfile=tmp_path / "theta90", hide_progress_bars=True,
        )
        dpw = captured["grid"].discreteplanewaves[0]
        assert list(dpw.m[:3]) == [2, 1, 0]
    finally:
        model_mod.Model.build = orig_build


# --- Solver-level physics -------------------------------------------------

_DEAD = {"TM": ("Ex", "Ey", "Hz"), "TE": ("Ez", "Hx", "Hy")}
_LIVE = {"TM": ("Ez", "Hx", "Hy"), "TE": ("Ex", "Ey", "Hz")}


@pytest.mark.parametrize("mode,psi", [("TM", 90), ("TE", 0)])
def test_2d_dpw_solve_dead_components_identically_zero(tmp_path, mode, psi):
    """Direct test of the skip_axis fix: a full solve must leave the mode's
    structurally-dead components identically zero at a receiver inside the
    TFSF box, while the live components carry a real propagating wave.
    Fails loudly without the invariant-axis face skip."""
    scene = _scene_2d(mode)
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=psi))
    scene.add(gprMax.Rx(p1=(0.03, 0.03, INF)))

    _run(scene, tmp_path, f"solve_{mode}", geometry_only=False)

    with h5py.File(str(tmp_path / f"solve_{mode}") + ".h5", "r") as f:
        rx = f["rxs/rx1"]
        for comp in _DEAD[mode]:
            data = rx[comp][:]
            assert np.all(data == 0), f"dead component {comp} is nonzero (max {np.max(np.abs(data)):g})"
        live_max = max(np.max(np.abs(rx[comp][:])) for comp in _LIVE[mode])
        assert live_max > 1e-3, "live components degenerate (wave never arrived)"
        for comp in _LIVE[mode]:
            assert not np.any(np.isnan(rx[comp][:])), f"{comp} contains NaN"


def test_2d_dpw_tfsf_leakage_outside_box_is_small(tmp_path):
    """The 4-edge 2D TFSF must close: a receiver OUTSIDE the box sees only
    numerical leakage, orders of magnitude below the wave inside."""
    scene = _scene_2d("TM")
    scene.add(_angles_dpw(theta=90, phi=26.565051177, psi=90))
    scene.add(gprMax.Rx(p1=(0.03, 0.03, INF), id="inside"))
    scene.add(gprMax.Rx(p1=(0.008, 0.008, INF), id="outside"))

    _run(scene, tmp_path, "leakage", geometry_only=False)

    with h5py.File(str(tmp_path / "leakage") + ".h5", "r") as f:
        inside = np.max(np.abs(f["rxs/rx1/Ez"][:]))
        outside = np.max(np.abs(f["rxs/rx2/Ez"][:]))
    assert inside > 1e-3
    assert outside < 1e-3 * inside, (
        f"TFSF leakage too large: outside {outside:g} vs inside {inside:g}"
    )


def test_axial_2d_solve_dead_components_identically_zero(tmp_path):
    """Axial (previously crashed outright in 2D via out-of-bounds
    G.ID[0,2,2,2] indexing): must now run and keep dead components zero."""
    scene = _scene_2d("TM")
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.015, 0.015, INF), p2=(0.045, 0.045, INF),
            axis="x", psi=90, waveform_id="w",
        )
    )
    scene.add(gprMax.Rx(p1=(0.03, 0.03, INF)))

    _run(scene, tmp_path, "axial_solve", geometry_only=False)

    with h5py.File(str(tmp_path / "axial_solve") + ".h5", "r") as f:
        rx = f["rxs/rx1"]
        for comp in ("Ex", "Ey", "Hz"):
            assert np.all(rx[comp][:] == 0), f"dead component {comp} is nonzero"
        assert np.max(np.abs(rx["Ez"][:])) > 1e-3
