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

"""End-only taper semantics, public inputs, and saved-data compatibility."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.ntff.conventions import engineering_dft
from gprMax.ntff.frequency_domain import _window
from gprMax.ntff.surfaces import build_component_surface
from .test_frequency_domain import _configured_monitor, _observe_ex_monitor


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("count", (1, 2, 6, 11, 101, 102))
def test_end_taper_has_flat_start_and_correct_half_cosine(dtype, count):
    weights = _window("hann_end", count, dtype)
    assert weights.dtype == dtype
    assert not weights.flags.writeable
    assert weights[0] == 1
    assert np.all(np.diff(weights) <= 0)
    assert np.all((weights >= 0) & (weights <= 1))
    if count == 1:
        return
    position = np.arange(count) / (count - 1)
    np.testing.assert_array_equal(weights[position <= 0.8], 1)
    expected = np.where(position <= 0.8, 1, np.cos(np.pi * (position - 0.8) / 0.4) ** 2)
    np.testing.assert_allclose(weights, expected, rtol=2e-6, atol=2e-7 if dtype == np.float32 else 1e-15)
    assert weights[-1] == 0


@pytest.mark.parametrize("name", ("rectangular", "hann"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_existing_windows_are_bit_identical(name, dtype):
    expected = np.ones(101, dtype=dtype) if name == "rectangular" else np.hanning(101).astype(dtype)
    np.testing.assert_array_equal(_window(name, 101, dtype), expected)


def test_early_delayed_impulse_ratio_is_preserved_but_late_signal_is_not():
    count = 101
    incident = np.zeros(count)
    scattered = np.zeros(count)
    incident[10], scattered[20] = 1, 2
    frequencies = [1.0, 5.0, 10.0]

    def ratio(window):
        return np.abs(
            engineering_dft(scattered, frequencies, 0.001, window=window)
            / engineering_dft(incident, frequencies, 0.001, window=window)
        )

    np.testing.assert_allclose(ratio(_window("hann_end", count, np.float64)), 2)
    assert np.all(ratio(_window("hann", count, np.float64)) > 6)
    scattered[:] = 0
    scattered[90] = 2
    # No promise that an end taper preserves physical late-time ringing.
    np.testing.assert_allclose(ratio(_window("hann_end", count, np.float64)), 1)


@pytest.mark.parametrize("other_window", ("rectangular", "hann"))
def test_saved_incident_subtraction_rejects_other_window(tmp_path, other_window):
    surface = build_component_surface("Ex", (2, 2, 2), (5, 5, 5), (0.02,) * 3, (9, 9, 9))
    reference = _configured_monitor("Ex", surface, [1e9], 1e-11, 12, window="hann_end")
    _observe_ex_monitor(reference, surface, 12, 1e9)
    path = tmp_path / "reference.h5"
    with h5py.File(path, "w") as output:
        reference.write_hdf5(output)
    target = _configured_monitor(
        "Ex",
        surface,
        [1e9],
        1e-11,
        12,
        window=other_window,
        incident_surface_file=path,
        incident_monitor_name="monitor",
    )
    with pytest.raises(ValueError, match="incompatible"):
        _observe_ex_monitor(target, surface, 12, 1e9)


def test_hash_transforms_share_end_taper_and_store_metadata(tmp_path):
    path = tmp_path / "windows.in"
    path.write_text(
        """#domain: .08 .08 .08
#dx_dy_dz: .004 .004 .004
#time_window: 2e-10
#pml_cells: 3
#num_threads: 1
#waveform: ricker 1 5e9 pulse
#hertzian_dipole: z .04 .04 .04 pulse
#ntff_surface: .028 .028 .028 .052 .052 .052 s
#ntff_layered_background: bg z free_space
#ksir_frequency: s ks 5e9 hann_end
#ntff_frequency: s eq 5e9 hann_end
#ntff_layered_frequency: s layer bg 5e9 hann_end
#ntff_far_field: 45 20 eq far Etheta
#ntff_far_field: 45 20 layer farlayer Etheta
"""
    )
    gprMax.run(inputfile=path, hide_progress_bars=True, cpu_precision="double")
    with h5py.File(path.with_suffix(".h5")) as output:
        for transform in ("ks", "eq", "layer"):
            group = output[f"ntff/s/frequency/{transform}"]
            assert group.attrs["window"] == "hann_end"
            assert group.attrs["window_taper_fraction"] == 0.2
        np.testing.assert_allclose(
            output["ntff/s/frequency/eq/far_field/far/fields/Etheta"][...],
            output["ntff/s/frequency/layer/far_field/farlayer/fields/Etheta"][...],
            rtol=1e-10,
            atol=1e-25,
        )
