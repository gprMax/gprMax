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

"""Tests for #magnetic_frill_source / MagneticFrillSource - the equivalent
feed model for an antenna driven through a PEC ground plane by a coaxial
line (Hyun, Kim & Kim, IEEE T-AP 57(1), 2009).

Covers: user-object parameter validation (fast, monkeypatched-grid unit
tests, mirroring tests/cmds_multiuse/test_metal_gpu_restriction_gaps.py's
style), finalise_setup()'s build-time geometry/symmetry/PML checks, a full
CPU solve with sane HDF5 output, PEC box/plate ground planes, symmetry-plane
placement, automatic port output, Hyun's terminal identity, and invariance of the
feed-cell self-admittance under symmetry completion.

The deeper physics validation ladder (open/short/matched-load analytic
checks, the Maloney monopole benchmark) is a separate, follow-on
validation phase, not covered here.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.hash_cmds_file import get_user_objects
from gprMax.user_objects.cmds_multiuse import MagneticFrillSource

INF = float("inf")


# ---------------------------------------------------------------------------
# Fast parameter-validation unit tests (no grid build required)
# ---------------------------------------------------------------------------


def _set_solver(monkeypatch, solver="cpu", mpi=False, subgrid=False):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": solver, "subgrid": subgrid}
    config.sim_config.mpi = mpi
    config.sim_config.dtypes = {"float_or_double": np.float64}
    config.sim_config.em_consts = {"z0": 376.730313668}
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))


def _fake_grid(dx=1e-3, dy=1e-3, dz=1e-3, waveform_id="w"):
    return SimpleNamespace(dx=dx, dy=dy, dz=dz, waveforms=[SimpleNamespace(ID=waveform_id)])


def _frill(**overrides):
    kwargs = dict(p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w")
    kwargs.update(overrides)
    return MagneticFrillSource(**kwargs)


@pytest.mark.parametrize(
    "command, expected_polarisation, expected_zcoax, expected_start, expected_stop",
    [
        ("#magnetic_frill_source: z 0.01 0.02 0.03 50 w", "z", 50, None, None),
        (
            "#magnetic_frill_source: x 0.01 0.02 0.03 75 w 1e-10 2e-10",
            "x",
            75,
            1e-10,
            2e-10,
        ),
    ],
)
def test_hash_command_builds_api_object(
    command, expected_polarisation, expected_zcoax, expected_start, expected_stop
):
    objects = get_user_objects([f"{command}\n"], checkessential=False)

    assert len(objects) == 1
    frill = objects[0]
    assert isinstance(frill, MagneticFrillSource)
    assert frill.point == (0.01, 0.02, 0.03)
    assert frill.polarisation == expected_polarisation
    assert frill.zcoax == expected_zcoax
    assert frill.waveform_id == "w"
    assert frill.start == expected_start
    assert frill.stop == expected_stop


@pytest.mark.parametrize(
    "command",
    [
        "#magnetic_frill_source: z 0.01 0.02 0.03 50",
        "#magnetic_frill_source: z 0.01 0.02 0.03 50 w 1e-10 2e-10 10 extra",
    ],
)
def test_hash_command_rejects_invalid_parameter_count(command):
    with pytest.raises(ValueError, match="requires six parameters"):
        get_user_objects([f"{command}\n"], checkessential=False)


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
def test_accelerator_solvers_are_accepted(monkeypatch, solver):
    _set_solver(monkeypatch, solver)
    _frill()._validate_parameters(_fake_grid())


def test_allowed_with_mpi(monkeypatch):
    _set_solver(monkeypatch, "cpu", mpi=True)
    _frill()._validate_parameters(_fake_grid())


def test_main_grid_frill_is_accepted_when_model_contains_subgrid(monkeypatch):
    _set_solver(monkeypatch, "cpu", subgrid=True)
    _frill()._validate_parameters(_fake_grid())


def test_rejected_in_2d_mode(monkeypatch):
    _set_solver(monkeypatch)
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="2D TM"))
    with pytest.raises(ValueError, match="2D mode"):
        _frill()._validate_parameters(_fake_grid())


@pytest.mark.parametrize("polarisation", ["x", "y", "Z", "w"])
def test_polarisation_restriction(monkeypatch, polarisation):
    _set_solver(monkeypatch)
    frill = _frill(polarisation=polarisation)
    if polarisation in ("x", "y", "Z"):
        frill._validate_parameters(_fake_grid())  # x/y/z allowed, case-insensitive
        assert frill.polarisation == polarisation.lower()
    else:
        with pytest.raises(ValueError, match="polarisation must be x, y, or z"):
            frill._validate_parameters(_fake_grid())


@pytest.mark.parametrize("zcoax", [0, -50, float("nan"), float("inf")])
def test_rejects_invalid_zcoax(monkeypatch, zcoax):
    _set_solver(monkeypatch)
    with pytest.raises(ValueError, match="zcoax"):
        _frill(zcoax=zcoax)._validate_parameters(_fake_grid())


def test_rejects_unknown_waveform(monkeypatch):
    _set_solver(monkeypatch)
    with pytest.raises(ValueError, match="no waveform"):
        _frill(waveform_id="missing")._validate_parameters(_fake_grid())


def test_rejects_negative_start(monkeypatch):
    _set_solver(monkeypatch)
    with pytest.raises(ValueError, match="less"):
        _frill(start=-1, stop=1)._validate_parameters(_fake_grid())


def test_rejects_zero_duration(monkeypatch):
    _set_solver(monkeypatch)
    with pytest.raises(ValueError, match="duration"):
        _frill(start=1, stop=1)._validate_parameters(_fake_grid())


# ---------------------------------------------------------------------------
# End-to-end build/solve tests
# ---------------------------------------------------------------------------


def _add_attached_wire(scene, p1, polarisation, length=0.01, radius=0.1e-3):
    p2 = list(p1)
    p2["xyz".index(polarisation)] += length
    scene.add(gprMax.ThinWire(p1=p1, p2=tuple(p2), radius=radius))


def _base_scene(
    domain=(0.02, 0.02, 0.02),
    dl=1e-3,
    time=2e-10,
    ground_plane=True,
    thin_wire=True,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=time))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    if ground_plane:
        scene.add(gprMax.Box(p1=(0, 0, 0), p2=(domain[0], domain[1], dl), material_id="pec"))
    if thin_wire:
        _add_attached_wire(scene, (0.01, 0.01, 0.0), "z")
    return scene


def test_rejects_missing_attached_thin_wire(tmp_path):
    scene = _base_scene(thin_wire=False)
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    with pytest.raises(ValueError, match="co-located #thin_wire"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=tmp_path / "no_thin_wire",
            hide_progress_bars=True,
        )


def test_rejects_overlapping_frill_stencils(tmp_path):
    scene = _base_scene()
    for _ in range(2):
        scene.add(
            gprMax.MagneticFrillSource(
                p1=(0.01, 0.01, 0.0),
                polarisation="z",
                zcoax=50,
                waveform_id="w",
            )
        )
    with pytest.raises(ValueError, match="overlapping magnetic feed edge"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "overlapping_frills",
            hide_progress_bars=True,
        )


def test_rejects_non_pec_feed_point(tmp_path):
    scene = _base_scene(ground_plane=False)
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    with pytest.raises(ValueError, match="PEC ground plane"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=tmp_path / "no_pec",
            hide_progress_bars=True,
        )


def test_rejects_domain_boundary_without_symmetry(tmp_path):
    scene = _base_scene()
    _add_attached_wire(scene, (0.0, 0.01, 0.0), "z")
    scene.add(
        gprMax.MagneticFrillSource(p1=(0.0, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w")
    )
    with pytest.raises(ValueError, match="PMC symmetry boundary"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=tmp_path / "no_symmetry",
            hide_progress_bars=True,
        )


def test_rejects_xmax_symmetry_corner(tmp_path):
    scene = _base_scene(domain=(0.02, 0.02, 0.02))
    scene.add(gprMax.SymmetryBoundary(face="xmax", type="pmc"))
    _add_attached_wire(scene, (0.02, 0.01, 0.0), "z")
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.02, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    with pytest.raises(ValueError, match="xmax-type symmetry corner"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=tmp_path / "xmax_corner",
            hide_progress_bars=True,
        )


def test_rejects_feed_point_inside_pml(tmp_path):
    scene = gprMax.Scene()
    dl = 1e-3
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.03, 0.03, dl), material_id="pec"))
    _add_attached_wire(scene, (0.005, 0.005, 0.0), "z")
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.005, 0.005, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    with pytest.raises(ValueError, match="PML"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=tmp_path / "in_pml",
            hide_progress_bars=True,
        )


def test_basic_solve_produces_finite_hdf5_output(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    outputfile = tmp_path / "frill_basic"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as f:
        assert f.attrs["nsrc"] == 1
        grp = f["frills/frill1"]
        for name in ("Vinc", "Vtotal", "Itot", "S11", "Zin", "Yin", "frequency"):
            assert name in grp
        vtot = grp["Vtotal"][:]
        itot = grp["Itot"][:]
        assert np.all(np.isfinite(vtot))
        assert np.all(np.isfinite(itot))
        assert np.max(np.abs(vtot)) > 0
        assert grp.attrs["Mirror1"] == False  # noqa: E712
        assert grp.attrs["Mirror2"] == False  # noqa: E712
        assert grp.attrs["InnerConductorRadius"] == pytest.approx(0.1e-3)
        assert grp.attrs["CurrentTimeApproximation"] == "average"
        assert grp.attrs["FeedSelfAdmittance"] > 0
        assert grp.attrs["TimeOffset"] == 0
        np.testing.assert_allclose(
            vtot,
            2 * grp["Vinc"][:] - grp.attrs["Z0"] * itot,
            rtol=2e-5,
            atol=5e-8,
        )


def test_pec_plate_ground_plane_is_accepted(tmp_path):
    scene = _base_scene(ground_plane=False)
    scene.add(gprMax.Plate(p1=(0, 0, 0), p2=(0.02, 0.02, 0), material_id="pec"))
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=50, waveform_id="w"
        )
    )
    outputfile = tmp_path / "frill_pec_plate"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as output:
        assert np.max(np.abs(output["frills/frill1/Vtotal"][:])) > 0


def test_symmetry_corner_placement_runs(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.SymmetryBoundary(face="y0", type="pmc"))
    _add_attached_wire(scene, (0.0, 0.0, 0.0), "z")
    scene.add(
        gprMax.MagneticFrillSource(p1=(0.0, 0.0, 0.0), polarisation="z", zcoax=50, waveform_id="w")
    )
    outputfile = tmp_path / "frill_corner"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as f:
        grp = f["frills/frill1"]
        assert grp.attrs["Mirror1"] == True  # noqa: E712
        assert grp.attrs["Mirror2"] == True  # noqa: E712
        vtot = grp["Vtotal"][:]
        assert np.all(np.isfinite(vtot))
        assert np.max(np.abs(vtot)) > 0


def test_symmetry_completion_preserves_feed_self_admittance(tmp_path):
    """An image doubles the current-loop weight, not the retained H deposit."""

    def _run(scene, p1, label):
        scene.add(gprMax.MagneticFrillSource(p1=p1, polarisation="z", zcoax=50, waveform_id="w"))
        outputfile = tmp_path / label
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=outputfile,
            hide_progress_bars=True,
        )
        with h5py.File(str(outputfile) + ".h5", "r") as output:
            return float(output["frills/frill1"].attrs["FeedSelfAdmittance"])

    full_scene = _base_scene()
    full_admittance = _run(full_scene, (0.01, 0.01, 0.0), "frill_full_g")

    corner_scene = _base_scene(thin_wire=False)
    corner_scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    corner_scene.add(gprMax.SymmetryBoundary(face="y0", type="pmc"))
    _add_attached_wire(corner_scene, (0.0, 0.0, 0.0), "z")
    corner_admittance = _run(corner_scene, (0.0, 0.0, 0.0), "frill_corner_g")

    assert corner_admittance == pytest.approx(full_admittance)


def test_x_polarisation_basic_solve_produces_finite_output(tmp_path):
    """x-polarisation is not a theoretical restriction lifted for free - it
    uses a genuinely different pair of transverse H components (Hy/Hz
    instead of z-polarisation's Hy/Hx), so this needs its own end-to-end
    check, not just trusting the z-polarisation derivation by symmetry."""
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    # Ground plane perpendicular to x (the polarisation axis): a thin slab
    # spanning the full y-z extent at x in [0, dl).
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(dl, domain[1], domain[2]), material_id="pec"))
    _add_attached_wire(scene, (0.0, 0.01, 0.01), "x")
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.0, 0.01, 0.01), polarisation="x", zcoax=50, waveform_id="w"
        )
    )
    outputfile = tmp_path / "frill_x_pol"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as f:
        grp = f["frills/frill1"]
        assert grp.attrs["Polarisation"] == "x"
        vtot = grp["Vtotal"][:]
        itot = grp["Itot"][:]
        assert np.all(np.isfinite(vtot))
        assert np.all(np.isfinite(itot))
        assert np.max(np.abs(vtot)) > 0


def test_y_polarisation_basic_solve_produces_finite_output(tmp_path):
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    # Ground plane perpendicular to y: a thin slab spanning the full x-z
    # extent at y in [0, dl).
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(domain[0], dl, domain[2]), material_id="pec"))
    _add_attached_wire(scene, (0.01, 0.0, 0.01), "y")
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.0, 0.01), polarisation="y", zcoax=50, waveform_id="w"
        )
    )
    outputfile = tmp_path / "frill_y_pol"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as f:
        grp = f["frills/frill1"]
        assert grp.attrs["Polarisation"] == "y"
        vtot = grp["Vtotal"][:]
        itot = grp["Itot"][:]
        assert np.all(np.isfinite(vtot))
        assert np.all(np.isfinite(itot))
        assert np.max(np.abs(vtot)) > 0


@pytest.mark.parametrize(
    "polarisation,faces,p1",
    [
        ("x", ("y0", "z0"), (0.0, 0.0, 0.0)),
        ("y", ("z0", "x0"), (0.0, 0.0, 0.0)),
    ],
)
def test_symmetry_corner_placement_runs_for_x_and_y_polarisation(tmp_path, polarisation, faces, p1):
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    for face in faces:
        scene.add(gprMax.SymmetryBoundary(face=face, type="pmc"))
    ground_plane_p2 = {
        "x": (dl, domain[1], domain[2]),
        "y": (domain[0], dl, domain[2]),
    }[polarisation]
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=ground_plane_p2, material_id="pec"))
    _add_attached_wire(scene, p1, polarisation)
    scene.add(
        gprMax.MagneticFrillSource(p1=p1, polarisation=polarisation, zcoax=50, waveform_id="w")
    )
    outputfile = tmp_path / f"frill_corner_{polarisation}"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as f:
        grp = f["frills/frill1"]
        assert grp.attrs["Mirror1"] == True  # noqa: E712
        assert grp.attrs["Mirror2"] == True  # noqa: E712
        vtot = grp["Vtotal"][:]
        assert np.all(np.isfinite(vtot))
        assert np.max(np.abs(vtot)) > 0


def test_zcoax_changes_reflection(tmp_path):
    """Sanity check that Z0 (now supplied directly as zcoax) actually
    participates in the solve rather than the source being a dead/no-op
    stencil: two otherwise identical models differing only in zcoax must
    produce different Vtotal histories.
    """

    def _run(zcoax, label):
        scene = _base_scene()
        scene.add(
            gprMax.MagneticFrillSource(
                p1=(0.01, 0.01, 0.0), polarisation="z", zcoax=zcoax, waveform_id="w"
            )
        )
        outputfile = tmp_path / label
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=False,
            outputfile=outputfile,
            hide_progress_bars=True,
        )
        with h5py.File(str(outputfile) + ".h5", "r") as f:
            return f["frills/frill1/Vtotal"][:].copy(), f["frills/frill1"].attrs["Z0"]

    vtot_a, z0_a = _run(30, "frill_zcoax_small")
    vtot_b, z0_b = _run(90, "frill_zcoax_large")

    assert z0_a != z0_b
    assert not np.allclose(vtot_a, vtot_b)


def test_source_sets_automatic_port_spectrum_limit(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0),
            polarisation="z",
            zcoax=50,
            waveform_id="w",
            spectrum_limit=5,
        )
    )
    outputfile = tmp_path / "frill_spectrum_limit"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as f:
        grp = f["frills/frill1"]
        assert grp.attrs["MinimumWavelengthCells"] == 5


def test_frill_automatic_port_keeps_fixed_identifier(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.01, 0.01, 0.0),
            polarisation="z",
            zcoax=50,
            waveform_id="w",
            spectrum_limit=5,
        )
    )
    outputfile = tmp_path / "frill_port_id"
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=False,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    with h5py.File(str(outputfile) + ".h5", "r") as f:
        assert "frills/frill1" in f
