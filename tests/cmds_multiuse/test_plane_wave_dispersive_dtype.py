# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Regression tests for dispersive discrete-plane-wave auxiliary state.

Lorentz or Drude materials make the model-wide dispersive coefficient array
complex, including otherwise-real Debye rows. The DPW must consequently use
the same real/complex auxiliary-state dtype as the main grid and apply the
same real-field projection of the complex pole recurrence. These tests cover
the original mixed-material dtype failure and direct, multipole Lorentz/Drude
propagation in both homogeneous and axial DPWs.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.cython.plane_wave import updatePlaneWave_electric_dispersive


def _add_dispersion(scene, kind, material_id, poles=1):
    if kind == "debye":
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=poles,
                er_delta=[2.0, 0.75][:poles],
                tau=[1e-11, 2e-11][:poles],
                material_ids=[material_id],
            )
        )
    elif kind == "lorentz":
        scene.add(
            gprMax.AddLorentzDispersion(
                poles=poles,
                er_delta=[1.0, 0.5][:poles],
                omega=[8e9, 15e9][:poles],
                delta=[1e9, 1.5e9][:poles],
                material_ids=[material_id],
            )
        )
    elif kind == "drude":
        scene.add(
            gprMax.AddDrudeDispersion(
                poles=poles,
                # Keep epsilon positive around the 10 GHz source while still
                # exercising non-negligible, multi-pole Drude dispersion.
                omega=[1e9, 2e9][:poles],
                alpha=[0.8e9, 1.2e9][:poles],
                material_ids=[material_id],
            )
        )
    else:
        raise ValueError(f"Unknown dispersion type: {kind}")


def _debye_plane_wave_scene(extra_dispersion=None):
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(20e-3, 20e-3, 20e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=5e-11))

    scene.add(gprMax.Material(er=2, se=0, mr=1, sm=0, id="debye_bg"))
    _add_dispersion(scene, "debye", "debye_bg")
    if extra_dispersion in ("lorentz", "drude"):
        scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="unused"))
        _add_dispersion(scene, extra_dispersion, "unused")

    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(20e-3, 20e-3, 20e-3),
            material_id="debye_bg",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(4e-3, 4e-3, 4e-3),
            p2=(16e-3, 16e-3, 16e-3),
            m_vec=(1, 0, 0),
            psi=0,
            waveform_id="w",
            material_id="debye_bg",
        )
    )
    scene.add(gprMax.Rx(p1=(10e-3, 10e-3, 10e-3)))
    return scene


def _lorentz_drude_plane_wave_scene(kind, axial):
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(20e-3, 20e-3, 20e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=3e-10))

    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="dispersive_bg"))
    _add_dispersion(scene, kind, "dispersive_bg", poles=2)
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(20e-3, 20e-3, 20e-3),
            material_id="dispersive_bg",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    if axial:
        scene.add(
            gprMax.DiscretePlaneWaveAxial(
                p1=(4e-3, 4e-3, 4e-3),
                p2=(16e-3, 16e-3, 16e-3),
                axis="x",
                psi=0,
                waveform_id="w",
            )
        )
    else:
        scene.add(
            gprMax.DiscretePlaneWaveVector(
                p1=(4e-3, 4e-3, 4e-3),
                p2=(16e-3, 16e-3, 16e-3),
                m_vec=(1, 0, 0),
                psi=0,
                waveform_id="w",
                material_id="dispersive_bg",
            )
        )
    scene.add(gprMax.Rx(p1=(10e-3, 10e-3, 10e-3)))
    return scene


def _free_space_plane_wave_scene(extra_dispersion=None):
    """Free-space DPW in a model whose global pole storage may be complex."""

    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(20e-3,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=5e-11))
    if extra_dispersion in ("lorentz", "drude"):
        scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="unused"))
        _add_dispersion(scene, extra_dispersion, "unused")
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(4e-3,) * 3,
            p2=(16e-3,) * 3,
            m_vec=(1, 0, 0),
            psi=0,
            waveform_id="w",
            material_id="free_space",
        )
    )
    scene.add(gprMax.Rx(p1=(10e-3,) * 3))
    return scene


def _run_trace(tmp_path, name, extra_dispersion=None):
    output = Path(tmp_path, name)
    gprMax.run(
        scenes=[_debye_plane_wave_scene(extra_dispersion)],
        outputfile=output,
        hide_progress_bars=True,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as h5:
        return np.asarray(h5["rxs/rx1/Ey"])


def test_homogeneous_complex_pole_current_uses_complete_product():
    """The homogeneous DPW must retain the Lorentz imaginary cross term."""

    n = 3
    pml_cells = 0
    auxiliary_fields = np.zeros((3, n), dtype=np.float64)
    magnetic_fields = np.zeros_like(auxiliary_fields)
    px = np.zeros((1, n), dtype=np.complex128)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    px[0, 1] = 3 + 4j
    pml_integrals = np.zeros((2, pml_cells), dtype=np.float64)
    pml_coefficients = np.zeros((4, pml_cells), dtype=np.float64)
    updatecoeffs_e = np.array([1, 0, 0, 0, 1], dtype=np.float64)
    updatecoeffs_h = np.zeros(5, dtype=np.float64)
    updatecoeffs_dispersive = np.array([1 + 2j, 1, 0], dtype=np.complex128)
    main_fields = np.zeros((1, 1, 1), dtype=np.float64)
    projections = np.zeros(6, dtype=np.float64)
    waveform_values = np.zeros((2, 3, 1), dtype=np.float64)
    mapping = np.array([1, 0, 0, 1], dtype=np.int32)
    origin = np.zeros(3, dtype=np.int32)
    corners = np.zeros(6, dtype=np.int32)
    owned_lower = np.zeros(3, dtype=np.int32)
    owned_upper = np.ones(3, dtype=np.int32)

    updatePlaneWave_electric_dispersive(
        n,
        pml_cells,
        1,
        0,
        magnetic_fields,
        auxiliary_fields,
        px,
        py,
        pz,
        pml_integrals.copy(),
        pml_integrals.copy(),
        pml_integrals.copy(),
        updatecoeffs_e,
        updatecoeffs_h,
        updatecoeffs_dispersive,
        1,
        pml_coefficients.copy(),
        pml_coefficients.copy(),
        pml_coefficients.copy(),
        pml_coefficients.copy(),
        pml_coefficients.copy(),
        pml_coefficients.copy(),
        main_fields.copy(),
        main_fields.copy(),
        main_fields.copy(),
        main_fields.copy(),
        main_fields.copy(),
        main_fields.copy(),
        projections,
        waveform_values,
        waveform_values,
        mapping,
        origin,
        corners,
        owned_lower,
        owned_upper,
        True,
        0,  # iteration
        1,  # dt
        1,  # dx
        1,  # dy
        1,  # dz
        1,  # ds
        1,  # c
        0,  # waveform start
        0,  # waveform stop
        1,  # waveform frequency
        b"ricker",
    )

    # Re((1 + 2j) * (3 + 4j)) = -5, hence E_new = -phi = +5.
    # The former Re(a) * Re(T) implementation produced -3.
    assert auxiliary_fields[0, 1] == 5


@pytest.mark.parametrize("extra_dispersion", ["lorentz", "drude"])
def test_debye_plane_wave_accepts_model_wide_complex_storage(tmp_path, extra_dispersion):
    real_storage = _run_trace(tmp_path, "debye_only")
    complex_storage = _run_trace(
        tmp_path, f"debye_plus_{extra_dispersion}", extra_dispersion
    )

    assert np.max(np.abs(real_storage)) > 0
    assert np.all(np.isfinite(complex_storage))
    np.testing.assert_allclose(complex_storage, real_storage, rtol=1e-6, atol=1e-7)


def test_debye_plane_wave_double_precision(tmp_path):
    output = Path(tmp_path, "debye_double")
    gprMax.run(
        scenes=[_debye_plane_wave_scene()],
        outputfile=output,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as h5:
        trace = np.asarray(h5["rxs/rx1/Ey"])

    assert trace.dtype == np.float64
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0


@pytest.mark.parametrize("extra_dispersion", ["lorentz", "drude"])
def test_free_space_plane_wave_keeps_simple_update_with_complex_main_grid(
    tmp_path, extra_dispersion
):
    reference_path = Path(tmp_path, "free_space_only")
    mixed_path = Path(tmp_path, f"free_space_plus_{extra_dispersion}")
    gprMax.run(
        scenes=[_free_space_plane_wave_scene()],
        outputfile=reference_path,
        hide_progress_bars=True,
    )
    gprMax.run(
        scenes=[_free_space_plane_wave_scene(extra_dispersion)],
        outputfile=mixed_path,
        hide_progress_bars=True,
    )
    with h5py.File(reference_path.with_suffix(".h5"), "r") as reference_h5:
        reference = np.asarray(reference_h5["rxs/rx1/Ey"])
    with h5py.File(mixed_path.with_suffix(".h5"), "r") as mixed_h5:
        mixed = np.asarray(mixed_h5["rxs/rx1/Ey"])

    np.testing.assert_allclose(mixed, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kind", ["lorentz", "drude"])
@pytest.mark.parametrize("axial", [False, True], ids=["homogeneous", "axial"])
def test_multipole_lorentz_drude_plane_wave(tmp_path, kind, axial):
    output = Path(tmp_path, f"{kind}_{'axial' if axial else 'homogeneous'}")
    gprMax.run(
        scenes=[_lorentz_drude_plane_wave_scene(kind, axial)],
        outputfile=output,
        hide_progress_bars=True,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as h5:
        trace = np.asarray(h5["rxs/rx1/Ey"])

    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0


def test_multipole_lorentz_plane_wave_double_precision(tmp_path):
    output = Path(tmp_path, "lorentz_homogeneous_double")
    gprMax.run(
        scenes=[_lorentz_drude_plane_wave_scene("lorentz", axial=False)],
        outputfile=output,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as h5:
        trace = np.asarray(h5["rxs/rx1/Ey"])

    assert trace.dtype == np.float64
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0
