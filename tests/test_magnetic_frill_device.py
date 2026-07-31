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

"""Device-data and shared-template tests for magnetic-frill sources."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.cuda_opencl import knl_magnetic_frill_source
from gprMax.sources import (
    MAGNETIC_FRILL_MAX_TERMS,
    MagneticFrillSource,
    magnetic_frill_source_host_arrays,
)


@pytest.fixture
def float64_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float64}),
    )


def _frill(iterations=4):
    source = MagneticFrillSource(iterations, 0.1)
    source.ID = "frill"
    source.Z0 = 50.0
    source._G_coeff = 0.0125
    source._theta = 0.5
    source._previous_half_current = 0.025
    source._drive_terms = [
        ("Hx", 1, 2, 3, -0.4, 0.7),
        ("Hy", 2, 1, 3, 0.6, -0.8),
        ("Hz", 2, 3, 1, 0.5, 0.9),
    ]
    source.waveformvalues_wholedt = np.arange(iterations + 1, dtype=np.float64)
    return source


def test_host_arrays_pack_compiled_feed_terms(float64_config):
    source = _frill()
    arrays = magnetic_frill_source_host_arrays([source], SimpleNamespace(iterations=4))

    assert arrays["term_counts"].tolist() == [3]
    assert arrays["term_info"].shape == (1, MAGNETIC_FRILL_MAX_TERMS, 4)
    assert arrays["term_info"][0, :3].tolist() == [
        [0, 1, 2, 3],
        [1, 2, 1, 3],
        [2, 2, 3, 1],
    ]
    np.testing.assert_allclose(
        arrays["term_params"][0, :3],
        [[-0.4, 0.7], [0.6, -0.8], [0.5, 0.9]],
    )
    np.testing.assert_allclose(arrays["params"][0], [50.0, 0.0125, 0.5])
    assert arrays["state"].tolist() == [0.025]
    np.testing.assert_array_equal(arrays["waveform"][0], source.waveformvalues_wholedt)


def test_packed_recurrence_matches_cpu_update(float64_config):
    source = _frill()
    arrays = magnetic_frill_source_host_arrays([source], SimpleNamespace(iterations=4))
    fields_cpu = {
        "Hx": np.zeros((5, 5, 5), dtype=np.float64),
        "Hy": np.zeros((5, 5, 5), dtype=np.float64),
        "Hz": np.zeros((5, 5, 5), dtype=np.float64),
    }
    fields_cpu["Hx"][1, 2, 3] = 0.11
    fields_cpu["Hy"][2, 1, 3] = -0.07
    fields_cpu["Hz"][2, 3, 1] = 0.04
    fields_device = {name: values.copy() for name, values in fields_cpu.items()}

    source.update_magnetic(
        2,
        None,
        None,
        fields_cpu["Hx"],
        fields_cpu["Hy"],
        fields_cpu["Hz"],
        None,
    )

    current_bulk = 0.0
    for term in range(arrays["term_counts"][0]):
        component, x, y, z = arrays["term_info"][0, term]
        weight, _ = arrays["term_params"][0, term]
        field = ("Hx", "Hy", "Hz")[component]
        current_bulk += weight * fields_device[field][x, y, z]
    z0, admittance, theta = arrays["params"][0]
    previous = arrays["state"][0]
    vinc = 0.5 * arrays["waveform"][0, 2]
    zeta = admittance * z0
    current_new = (current_bulk + 2 * admittance * vinc - zeta * (1 - theta) * previous) / (
        1 + zeta * theta
    )
    current_centred = (1 - theta) * previous + theta * current_new
    vab = 2 * vinc - z0 * current_centred
    for term in range(arrays["term_counts"][0]):
        component, x, y, z = arrays["term_info"][0, term]
        _, source_gain = arrays["term_params"][0, term]
        field = ("Hx", "Hy", "Hz")[component]
        fields_device[field][x, y, z] += source_gain * vab

    assert source.Vinc[2] == pytest.approx(vinc)
    assert source.Itot[2] == pytest.approx(current_centred)
    assert source.Vtotal[2] == pytest.approx(vab)
    assert source._previous_half_current == pytest.approx(current_new)
    for component in fields_cpu:
        np.testing.assert_allclose(fields_cpu[component], fields_device[component])


def test_host_arrays_reject_too_many_feed_terms(float64_config):
    source = _frill()
    source._drive_terms *= 2

    with pytest.raises(ValueError, match="magnetic feed terms"):
        magnetic_frill_source_host_arrays([source], SimpleNamespace(iterations=4))


def test_host_arrays_guard_signed_32bit_kernel_indices(float64_config):
    with pytest.raises(ValueError, match="signed 32-bit index range"):
        magnetic_frill_source_host_arrays(
            [_frill()], SimpleNamespace(iterations=np.iinfo(np.int32).max)
        )


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_magnetic_frill_template_substitutions_are_complete(backend):
    kernel = knl_magnetic_frill_source.update_magnetic_frill_source
    arguments = kernel[f"args_{backend}"].substitute({"REAL": "double"})
    body = kernel["func"].substitute(
        {
            "CUDA_IDX": "int i = 0;" if backend == "cuda" else "",
            "REAL": "double",
            "MAX_FRILLTERMS": MAGNETIC_FRILL_MAX_TERMS,
            "NY_FRILLTERMINFO": 4,
            "NY_FRILLTERMPARAMS": 2,
            "NY_FRILLPARAMS": 3,
            "NY_FRILLWAVES": 5,
            "NY_FRILLOUT": 5,
        }
    )

    assert "$" not in arguments
    assert "$" not in body
    assert "first_active" not in body
    assert "frill_state[i] = current_new" in body
    assert "source_gain * V_ab" in body
