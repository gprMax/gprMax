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

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from gprMax.waveforms import Waveform as GprMaxWaveform
from toolboxes.ImpulseResponse import (
    BUILTIN_WAVEFORM_TYPES,
    SourceSampling,
    TargetWaveform,
    find_single_impulse,
    load_csv_waveforms,
    load_source_sampling,
    sample_builtin_waveform,
    synthesise_output,
    synthesise_receiver,
    waveform_energy_above,
    write_synthesised_output,
)
from toolboxes.SFCW.processing import SampledSignal, load_receiver, load_source


def _source(samples, dt=1e-10, time_offset=0.5e-10, evaluation_offset=None):
    signal = SampledSignal(
        path="/srcs/src1",
        samples=np.asarray(samples, dtype=np.float64),
        dt=dt,
        time_offset=time_offset,
        quantity="electric_current",
        units="A",
        source_type="HertzianDipole",
        filename="impulse.h5",
    )
    return SourceSampling(
        signal=signal,
        evaluation_time_offset=time_offset if evaluation_offset is None else evaluation_offset,
        update_lattice="electric_half_step",
        driving_quantity="electric_current",
    )


def _target(source, samples, waveform_id="target"):
    return TargetWaveform(
        id=waveform_id,
        samples=np.asarray(samples, dtype=np.float64),
        dt=source.signal.dt,
        source_time_offset=source.signal.time_offset,
        evaluation_time_offset=source.evaluation_time_offset,
        waveform_type="sampled",
    )


def test_find_single_impulse_reports_shift_and_amplitude():
    samples = np.zeros(32)
    samples[7] = -2.5
    assert find_single_impulse(samples) == (7, -2.5)
    with pytest.raises(ValueError, match="2 significant"):
        find_single_impulse(np.r_[1.0, 2.0, np.zeros(4)])


def test_shifted_impulse_is_removed_from_causal_convolution():
    count = 64
    impulse_index = 4
    amplitude = 2.0
    impulse = np.zeros(count)
    impulse[impulse_index] = amplitude
    source = _source(impulse)
    kernel = np.zeros(count - impulse_index)
    kernel[[1, 4, 9]] = (0.8, -0.3, 0.1)
    measured = np.zeros(count)
    measured[impulse_index:] = amplitude * kernel
    receiver = SampledSignal("/rxs/rx1/Ez", measured, source.signal.dt, 0.0, quantity="Ez")
    target_samples = np.zeros(count)
    target_samples[[0, 2, 5]] = (1.0, -0.25, 0.4)

    result, index, recovered_amplitude = synthesise_receiver(
        source,
        receiver,
        _target(source, target_samples),
    )

    expected = np.convolve(kernel, target_samples)[:count]
    assert index == impulse_index
    assert recovered_amplitude == amplitude
    assert_allclose(result.samples, expected, rtol=3e-14, atol=3e-14)


def test_bscan_traces_are_convolved_only_along_time_axis():
    count = 80
    impulse = np.r_[1.0, np.zeros(count - 1)]
    source = _source(impulse)
    receiver_samples = np.zeros((count, 3))
    receiver_samples[2, 0] = 1.0
    receiver_samples[5, 1] = -2.0
    receiver_samples[9, 2] = 0.25
    receiver = SampledSignal("/rxs/rx1/Ez", receiver_samples, source.signal.dt, 0.0, quantity="Ez")
    target_samples = np.zeros(count)
    target_samples[:3] = (1.0, 0.5, -0.2)

    result, _, _ = synthesise_receiver(source, receiver, _target(source, target_samples))

    assert result.samples.shape == receiver_samples.shape
    for trace in range(3):
        assert_allclose(
            result.samples[:, trace],
            np.convolve(receiver_samples[:, trace], target_samples)[:count],
            rtol=3e-14,
            atol=3e-14,
        )


def test_builtin_ricker_uses_half_step_source_evaluation():
    source = _source(np.r_[1.0, np.zeros(127)], dt=1e-11, time_offset=0.5e-11)
    waveform = sample_builtin_waveform(source, "ricker", 1.7, 500e6, "ricker500")
    times = 0.5 * source.signal.dt + source.signal.dt * np.arange(128)
    zeta = np.pi**2 * (500e6) ** 2
    delay = times - np.sqrt(2) / 500e6
    expected = (
        -1.7 * (2 * zeta * (2 * zeta * delay**2 - 1) * np.exp(-zeta * delay**2)) / (2 * zeta)
    )
    assert_allclose(waveform.samples, expected, rtol=2e-15, atol=2e-15)


def test_builtin_stop_time_zeros_later_updates():
    source = _source(np.r_[1.0, np.zeros(31)], dt=1e-10, time_offset=0.5e-10)
    waveform = sample_builtin_waveform(
        source,
        "contsine",
        1.0,
        500e6,
        "finite_sine",
        start_time=2e-10,
        stop_time=8e-10,
    )

    assert np.all(waveform.samples[:2] == 0)
    assert np.any(waveform.samples[2:9] != 0)
    assert np.all(waveform.samples[9:] == 0)


@pytest.mark.parametrize("waveform_type", BUILTIN_WAVEFORM_TYPES)
def test_builtin_samples_match_gprmax_waveform_definition(waveform_type):
    source = _source(np.r_[1.0, np.zeros(63)], dt=1e-11, time_offset=0.5e-11)
    result = sample_builtin_waveform(source, waveform_type, 1.3, 500e6, "comparison")
    reference = GprMaxWaveform()
    reference.type = waveform_type
    reference.amp = 1.3
    reference.freq = 500e6
    times = source.evaluation_time_offset + source.signal.dt * np.arange(64)
    expected = np.asarray([reference.calculate_value(time, source.signal.dt) for time in times])
    assert_allclose(result.samples, expected, rtol=2e-13, atol=2e-13)


def test_hard_source_uses_evaluation_offset_not_physical_output_offset(tmp_path):
    output = tmp_path / "hard.h5"
    dt = 2e-11
    with h5py.File(output, "w") as file:
        file.attrs["dt"] = dt
        excitation = file.create_group("srcs/src1/excitation")
        excitation.attrs["SampleInterval"] = dt
        excitation.attrs["TimeSampleOffset"] = dt
        excitation.attrs["DrivingQuantity"] = "imposed_gap_voltage"
        excitation.attrs["UpdateLattice"] = "electric"
        excitation.create_dataset("samples", data=np.r_[1.0, np.zeros(15)])

    source = load_source_sampling(output)
    waveform = sample_builtin_waveform(source, "impulse", 1, 1, "impulse")

    assert source.signal.time_offset == dt
    assert source.evaluation_time_offset == 0
    assert_array_equal(np.flatnonzero(waveform.samples), [0])


def test_csv_waveforms_are_resampled_on_source_evaluation_times(tmp_path):
    source = _source(np.r_[1.0, np.zeros(7)], dt=1.0, time_offset=0.5)
    csv = tmp_path / "pulses.csv"
    csv.write_text("time,pulse_a,pulse_b\n0,0,1\n1,2,0\n2,0,-1\n")

    first, second = load_csv_waveforms(csv, source)

    assert first.id == "pulse_a"
    assert_allclose(first.samples[:3], (1.0, 1.0, 0.0))
    assert_allclose(second.samples[:3], (0.5, -0.5, 0.0))
    assert np.all(first.samples[3:] == 0)


def test_waveform_energy_limit_reports_out_of_band_fraction():
    count = 128
    dt = 1e-3
    time = dt * np.arange(count)
    samples = np.sin(2 * np.pi * 62.5 * time) + np.sin(2 * np.pi * 125 * time)
    fraction = waveform_energy_above(samples, dt, 80.0)
    assert fraction == pytest.approx(0.5, abs=2e-14)


def _write_impulse_h5(path: Path, *, second_active_source=False):
    count = 64
    dt = 1e-10
    with h5py.File(path, "w") as output:
        output.attrs["dt"] = dt
        output.attrs["Iterations"] = count
        output.attrs["dx_dy_dz"] = (0.01, 0.01, 0.01)
        source = output.create_group("srcs/src1")
        source.attrs["Type"] = "HertzianDipole"
        excitation = source.create_group("excitation")
        excitation.attrs["SampleInterval"] = dt
        excitation.attrs["TimeSampleOffset"] = 0.5 * dt
        excitation.attrs["DrivingQuantity"] = "electric_current"
        excitation.attrs["Units"] = "A"
        excitation.attrs["UpdateLattice"] = "electric_half_step"
        excitation.create_dataset("samples", data=np.r_[1.0, np.zeros(count - 1)])
        if second_active_source:
            other = output.create_group("srcs/src2/excitation")
            other.attrs["SampleInterval"] = dt
            other.attrs["TimeSampleOffset"] = 0.5 * dt
            other.create_dataset("samples", data=np.r_[1.0, np.zeros(count - 1)])
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Name"] = "response"
        samples = np.zeros(count)
        samples[[3, 8, 14]] = (0.8, -0.2, 0.05)
        field = receiver.create_dataset("Ez", data=samples)
        field.attrs["SampleInterval"] = dt
        field.attrs["TimeSampleOffset"] = 0.0
        field.attrs["Quantity"] = "Ez"


def test_batch_synthesis_writes_receiver_compatible_hdf5(tmp_path):
    impulse_file = tmp_path / "impulse.h5"
    _write_impulse_h5(impulse_file)
    source = load_source_sampling(impulse_file)
    waveform = sample_builtin_waveform(source, "ricker", 1.0, 300e6, "ricker300")
    result = synthesise_output(impulse_file, waveform)
    output = write_synthesised_output(tmp_path / "ricker.h5", result)

    stored_source = load_source(output)
    stored_receiver = load_receiver(output)
    assert_allclose(stored_source.samples, waveform.samples)
    assert_allclose(stored_receiver.samples, result.receivers[0].samples)
    with h5py.File(output, "r") as file:
        assert file.attrs["Format"] == "gprMax impulse-response waveform synthesis"
        assert file["/rxs/rx1/Ez"].attrs["SynthesisedFromImpulse"]
        assert_allclose(file["/impulse_reference/source_samples"], source.signal.samples)


def test_batch_synthesis_rejects_another_active_source(tmp_path):
    impulse_file = tmp_path / "multiple.h5"
    _write_impulse_h5(impulse_file, second_active_source=True)
    source = load_source_sampling(impulse_file, "/srcs/src1")
    waveform = sample_builtin_waveform(source, "ricker", 1.0, 300e6, "ricker300")
    with pytest.raises(ValueError, match="additional active sources"):
        synthesise_output(impulse_file, waveform, source_path="/srcs/src1")
