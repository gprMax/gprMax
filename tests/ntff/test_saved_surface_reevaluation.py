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

"""Reevaluating far fields from surface DFTs saved by the scene objects."""

import shutil

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
from gprMax.ntff.evaluator import project_cartesian_to_spherical

DL = 0.004
FREQUENCY = 5e9
THETA = (30.0, 60.0, 90.0, 120.0)
PHI = (0.0, 45.0, 90.0)
TRANSFORM_GROUP = "ntff/surface/frequency/spectrum"


def _run(tmp_path, *, symmetry=None, transform=gprMax.KSIRFrequencyTransform, save=True):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=6e-10))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=FREQUENCY, id="pulse"))
    if symmetry is None:
        dipole, lower = (0.04, 0.04, 0.04), (0.024, 0.024, 0.024)
    else:
        scene.add(gprMax.SymmetryBoundary(face="x0", type=symmetry))
        dipole, lower = (0.008, 0.04, 0.04), (0.0, 0.024, 0.024)
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=dipole, waveform_id="pulse"))
    scene.add(gprMax.NTFFSurface(p1=lower, p2=(0.056, 0.056, 0.056), id="surface"))
    scene.add(
        transform(
            surface_id="surface", id="spectrum", frequencies=(FREQUENCY,), save_surface_dft=save
        )
    )
    far_field = (
        gprMax.KSIRFarFieldArray
        if transform is gprMax.KSIRFrequencyTransform
        else gprMax.NTFFFarFieldArray
    )
    scene.add(
        far_field(
            theta_start=THETA[0],
            theta_stop=THETA[-1],
            theta_step=THETA[1] - THETA[0],
            phi_start=PHI[0],
            phi_stop=PHI[-1],
            phi_step=PHI[1] - PHI[0],
            transform_id="spectrum",
            id="pattern",
            outputs=("Etheta", "Ephi"),
        )
    )
    output = tmp_path / "saved_surface"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    return output.with_suffix(".h5")


def _spherical(saved):
    cartesian = np.stack([saved.range_normalized_fields[c] for c in ("Ex", "Ey", "Ez")], axis=-1)
    spherical = project_cartesian_to_spherical(cartesian, saved.theta, saved.phi, degrees=True)
    return spherical[..., 1], spherical[..., 2]


def _in_run(filename):
    with h5py.File(filename, "r") as f:
        group = f[f"{TRANSFORM_GROUP}/far_field/pattern"]
        return group["fields/Etheta"][:], group["fields/Ephi"][:]


def _assert_matches_run(filename, monitor_name="spectrum"):
    saved = gprMax.evaluate_saved_surface_dft(filename, monitor_name, THETA, PHI)
    etheta, ephi = _spherical(saved)
    run_etheta, run_ephi = _in_run(filename)
    scale = np.max(np.abs(run_etheta))
    assert scale > 0
    assert_allclose(etheta, run_etheta, rtol=1e-9, atol=1e-9 * scale)
    assert_allclose(ephi, run_ephi, rtol=1e-9, atol=1e-9 * scale)
    return saved


def test_scene_transform_reevaluates_to_the_in_run_far_field(tmp_path):
    filename = _run(tmp_path)
    saved = _assert_matches_run(filename)
    assert saved.closure == "closed"
    assert saved.mathematically_closed
    with h5py.File(filename, "r") as f:
        stored_origin = f["ntff/surface"].attrs["physical_origin"]
    assert_allclose(saved.origin, stored_origin)

    by_path = gprMax.evaluate_saved_surface_dft(filename, "surface/frequency/spectrum", THETA, PHI)
    assert_allclose(by_path.range_normalized_fields["Ez"], saved.range_normalized_fields["Ez"])

    # New directions and a moved phase centre: only the phase changes.
    moved = gprMax.evaluate_saved_surface_dft(
        filename, "spectrum", [90.0], [0.0], origin=saved.origin + [0.01, 0, 0]
    )
    assert moved.range_normalized_fields["Ez"].shape == (1, 1)
    index = THETA.index(90.0) * len(PHI) + PHI.index(0.0)
    assert_allclose(
        np.abs(moved.range_normalized_fields["Ez"][0, 0]),
        np.abs(saved.range_normalized_fields["Ez"][0, index]),
        rtol=1e-9,
    )


@pytest.mark.parametrize("boundary", ["pec", "pmc"])
def test_symmetry_completion_is_applied_from_the_surface_metadata(tmp_path, boundary):
    filename = _run(tmp_path, symmetry=boundary)
    saved = _assert_matches_run(filename)
    assert saved.closure == "symmetry"
    assert saved.mathematically_closed


def test_files_without_per_component_geometry_still_reevaluate(tmp_path):
    filename = _run(tmp_path)
    older = tmp_path / "older.h5"
    shutil.copy(filename, older)
    with h5py.File(older, "r+") as f:
        for component in f[f"{TRANSFORM_GROUP}/surface_dft"].values():
            for name in ("logical_lower", "logical_upper", "grid_spacing", "field_shape"):
                del component.attrs[name]
    _assert_matches_run(older)


def test_transform_without_saved_surface_is_reported(tmp_path):
    filename = _run(tmp_path, save=False)
    with pytest.raises(ValueError, match="save_surface_dft=False"):
        gprMax.evaluate_saved_surface_dft(filename, "spectrum", THETA, PHI)


def test_equivalent_current_transform_is_reported(tmp_path):
    filename = _run(tmp_path, transform=gprMax.NTFFFrequencyTransform)
    with pytest.raises(ValueError, match="only KSIR surface DFTs"):
        gprMax.evaluate_saved_surface_dft(filename, "spectrum", THETA, PHI)


def test_unknown_names_are_reported(tmp_path):
    filename = _run(tmp_path)
    with pytest.raises(ValueError, match="no NTFF monitor 'missing'"):
        gprMax.evaluate_saved_surface_dft(filename, "missing", THETA, PHI)
