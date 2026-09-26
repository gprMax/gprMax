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

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gprMax.toolboxes.SFCW.processing import (
    SampledSignal,
    direct_frequency_response,
    engineering_dft,
    homodyne_frequency_response,
    list_receivers,
    list_sources,
    load_receiver,
    load_source,
    process_output,
    reconstruct_time_response,
    tail_relative_db,
    write_sfcw_output,
)


def _brute_dft(samples, dt, frequencies, offset):
    times = offset + dt * np.arange(len(samples))
    return dt * np.exp(-2j * np.pi * frequencies[:, None] * times) @ samples


def test_engineering_dft_matches_explicit_sum_for_stepped_frequencies():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=127)
    dt = 2.5e-12
    frequencies = np.linspace(0.7e9, 4.3e9, 41)
    offset = 0.5 * dt

    result = engineering_dft(samples, dt, frequencies, time_offset=offset)

    assert_allclose(result, _brute_dft(samples, dt, frequencies, offset), rtol=2e-11, atol=1e-22)


def test_engineering_dft_matches_explicit_sum_for_arbitrary_frequencies():
    samples = np.linspace(-1, 1, 53)
    dt = 1.1e-11
    frequencies = np.asarray((0.2e9, 0.91e9, 1.7e9, 3.12e9))

    result = engineering_dft(samples, dt, frequencies, time_offset=-0.5 * dt)

    assert_allclose(
        result,
        _brute_dft(samples, dt, frequencies, -0.5 * dt),
        rtol=2e-13,
        atol=1e-22,
    )


def test_engineering_dft_transforms_each_bscan_trace_along_time_axis():
    samples = np.column_stack((np.arange(31), np.arange(31) ** 2))
    dt = 2e-11
    frequencies = np.linspace(0.1e9, 2e9, 17)

    result = engineering_dft(samples, dt, frequencies, time_offset=0.0)

    assert result.shape == (frequencies.size, 2)
    for trace in range(samples.shape[1]):
        assert_allclose(
            result[:, trace],
            _brute_dft(samples[:, trace], dt, frequencies, 0.0),
            rtol=2e-12,
            atol=1e-22,
        )


def test_direct_response_removes_electric_source_half_step():
    dt = 1e-11
    frequencies = np.linspace(0.5e9, 5e9, 37)
    source = SampledSignal("/srcs/src1", np.r_[1.0, np.zeros(127)], dt, 0.5 * dt)
    receiver = SampledSignal("/rxs/rx1/Ez", np.r_[0.0, 1.0, np.zeros(126)], dt, 0.0)

    result = direct_frequency_response(source, receiver, frequencies)

    expected = np.exp(-2j * np.pi * frequencies * 0.5 * dt)
    assert_allclose(result.response, expected, rtol=2e-11, atol=2e-11)


def test_homodyne_route_matches_direct_impulse_response():
    dt = 5e-11
    frequencies = np.linspace(0.25e9, 1.6e9, 15)
    source = SampledSignal("/srcs/src1", np.r_[2.0, np.zeros(255)], dt, 0.5 * dt)
    kernel = np.zeros(256)
    kernel[3] = 0.8
    kernel[8] = -0.2
    kernel[14] = 0.07
    receiver = SampledSignal("/rxs/rx1/Ez", 2.0 * kernel, dt, 0.0)

    direct = direct_frequency_response(source, receiver, frequencies)
    homodyne = homodyne_frequency_response(source, receiver, frequencies, cycles=10)

    assert_allclose(homodyne.response, direct.response, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("source_offset_factor", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("receiver_offset_factor", [-0.5, 0.0])
def test_homodyne_rejects_nyquist_but_direct_response_supports_it(
    source_offset_factor, receiver_offset_factor
):
    dt = 1e-10
    frequency = np.array([0.5 / dt])
    source_offset = source_offset_factor * dt
    receiver_offset = receiver_offset_factor * dt
    source = SampledSignal("/srcs/src1", np.r_[1.0, np.zeros(31)], dt, source_offset)
    receiver = SampledSignal("/rxs/rx1/Ez", np.r_[0.0, 1.0, np.zeros(30)], dt, receiver_offset)

    with pytest.raises(ValueError, match="strictly below.*Nyquist"):
        homodyne_frequency_response(source, receiver, frequency)

    direct = direct_frequency_response(source, receiver, frequency)
    expected = np.exp(-2j * np.pi * frequency * (dt + receiver_offset - source_offset))
    assert_allclose(direct.response, expected, atol=1e-14)


def test_reconstruction_places_uniform_delay_at_correct_time():
    count = 64
    df = 10e6
    frequencies = 100e6 + df * np.arange(count)
    delay_index = 11
    time_step = 1 / (count * df)
    delay = delay_index * time_step
    source = SampledSignal("source", np.ones(2), 1e-12, 0.0)
    receiver = SampledSignal("receiver", np.ones(2), 1e-12, 0.0)
    response = SimpleNamespace(
        frequency=frequencies,
        response=np.exp(-2j * np.pi * frequencies * delay),
    )
    # reconstruct_time_response only needs the two public arrays.
    _ = source, receiver

    result = reconstruct_time_response(response, window="rectangular")
    padded = reconstruct_time_response(response, window="rectangular", zero_pad_factor=4)
    shifted = reconstruct_time_response(
        response,
        window="rectangular",
        time_shift=3 * time_step,
    )

    assert np.argmax(np.abs(result.complex_envelope)) == delay_index
    assert result.time[delay_index] == delay
    assert result.real_bandpass[delay_index] > 0
    assert np.max(np.abs(result.complex_envelope)) == pytest.approx(1.0)
    assert np.max(np.abs(padded.complex_envelope)) == pytest.approx(1.0)
    assert np.argmax(np.abs(shifted.complex_envelope)) == delay_index + 3


def test_direct_response_and_reconstruction_preserve_bscan_trace_axis():
    dt = 1e-11
    frequencies = np.linspace(0.1e9, 3e9, 32)
    source = SampledSignal("/srcs/src1", np.r_[1.0, np.zeros(127)], dt, 0.5 * dt)
    receiver_samples = np.zeros((128, 3))
    receiver_samples[3, 0] = 1.0
    receiver_samples[6, 1] = 2.0
    receiver_samples[9, 2] = -0.5
    receiver = SampledSignal("/rxs/rx1/Ez", receiver_samples, dt, 0.0)

    frequency = direct_frequency_response(source, receiver, frequencies)
    time = reconstruct_time_response(frequency, window="rectangular")

    assert frequency.response.shape == (frequencies.size, 3)
    assert time.real_bandpass.shape == (frequencies.size, 3)
    assert tail_relative_db(receiver_samples) == float("-inf")


@pytest.mark.parametrize("ancestor_dt", [None, 2e-12, 1e-9])
def test_hdf5_loader_uses_stored_source_and_receiver_time_offsets(tmp_path, ancestor_dt):
    output = tmp_path / "model.h5"
    with h5py.File(output, "w") as file:
        if ancestor_dt is not None:
            file.attrs["dt"] = ancestor_dt
        source = file.create_group("srcs/src1")
        source.attrs["Type"] = "HertzianDipole"
        excitation = source.create_group("excitation")
        excitation.attrs["SampleInterval"] = 2e-12
        excitation.attrs["TimeSampleOffset"] = 1e-12
        excitation.attrs["DrivingQuantity"] = "electric_current"
        excitation.attrs["Units"] = "A"
        excitation.attrs["SourceType"] = "HertzianDipole"
        excitation.attrs["SpatialScale"] = 0.001
        excitation.create_dataset("samples", data=np.r_[1.0, np.zeros(7)])
        receiver = file.create_group("rxs/rx1")
        field = receiver.create_dataset("Ez", data=np.arange(8, dtype=np.float32))
        field.attrs["SampleInterval"] = 2e-12
        field.attrs["TimeSampleOffset"] = 0.0

    assert list_sources(output) == ["/srcs/src1"]
    assert list_receivers(output) == {"/rxs/rx1": ("Ez",)}
    loaded_source = load_source(output)
    loaded_receiver = load_receiver(output)
    assert loaded_source.dt == 2e-12
    assert loaded_receiver.dt == 2e-12
    assert loaded_source.time_offset == 1e-12
    assert loaded_source.spatial_scale == 0.001
    assert loaded_receiver.time_offset == 0.0
    assert loaded_receiver.path == "/rxs/rx1/Ez"


@pytest.mark.parametrize("grid_path", ["/", "subgrids/fine"])
def test_hdf5_loader_falls_back_to_nearest_grid_interval(tmp_path, grid_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as output:
        output.attrs["dt"] = 1e-9
        grid = output if grid_path == "/" else output.create_group(grid_path)
        grid.attrs["dt"] = 2e-12
        source = grid.create_group("srcs/src1/excitation")
        source.attrs["TimeSampleOffset"] = 1e-12
        source.create_dataset("samples", data=np.r_[1.0, np.zeros(7)])
        grid.create_dataset("rxs/rx1/Hy", data=np.arange(8))

    source = load_source(path)
    receiver = load_receiver(path)

    assert source.dt == 2e-12
    assert receiver.dt == 2e-12
    assert receiver.time_offset == -1e-12


def test_merged_bscan_can_use_source_history_from_original_ascan(tmp_path):
    source_file = tmp_path / "scan1.h5"
    receiver_file = tmp_path / "scan_merged.h5"
    dt = 2e-11
    with h5py.File(source_file, "w") as output:
        output.attrs["dt"] = dt
        source = output.create_group("srcs/src1/excitation")
        source.attrs["SampleInterval"] = dt
        source.attrs["TimeSampleOffset"] = 0.5 * dt
        source.attrs["SourceType"] = "HertzianDipole"
        source.create_dataset("samples", data=np.r_[1.0, np.zeros(63)])
    with h5py.File(receiver_file, "w") as output:
        output.attrs["dt"] = dt
        receiver = output.create_group("rxs/rx1")
        field = receiver.create_dataset("Ez", data=np.zeros((64, 2)))
        field[3, 0] = 1.0
        field[7, 1] = -0.25
        field.attrs["SampleInterval"] = dt
        field.attrs["TimeSampleOffset"] = 0.0

    result = process_output(
        receiver_file,
        np.linspace(0.1e9, 2e9, 20),
        source_filename=source_file,
    )

    assert result.response.shape == (20, 2)
    assert result.source.filename == str(source_file)
    assert result.receiver.filename == str(receiver_file)


@pytest.mark.parametrize("input_role", ["source", "receiver"])
@pytest.mark.parametrize("alias_kind", ["same_path", "symlink", "hardlink"])
def test_sfcw_output_cannot_overwrite_an_input(tmp_path, input_role, alias_kind):
    files = {role: tmp_path / f"{role}.h5" for role in ("source", "receiver")}
    for role, path in files.items():
        with h5py.File(path, "w") as output:
            output.attrs["original_input"] = role
    before = {role: path.read_bytes() for role, path in files.items()}
    source = SampledSignal(
        "/srcs/src1",
        np.r_[1.0, np.zeros(31)],
        1e-10,
        0.0,
        filename=str(files["source"]),
    )
    receiver = SampledSignal(
        "/rxs/rx1/Ez",
        np.r_[0.0, 1.0, np.zeros(30)],
        1e-10,
        0.0,
        filename=str(files["receiver"]),
    )
    response = direct_frequency_response(source, receiver, [1e8, 2e8, 3e8])
    destination = files[input_role]
    if alias_kind != "same_path":
        destination = tmp_path / "alias.h5"
        if alias_kind == "symlink":
            try:
                destination.symlink_to(files[input_role])
            except OSError as error:
                pytest.skip(f"Filesystem does not permit symlink creation: {error}")
        else:
            destination.hardlink_to(files[input_role])

    with pytest.raises(ValueError, match="must not overwrite an input"):
        write_sfcw_output(destination, response)

    for role, path in files.items():
        assert path.read_bytes() == before[role]
    processed = tmp_path / "processed.h5"
    write_sfcw_output(processed, response)
    with h5py.File(processed, "r") as output:
        assert_allclose(output["response"], response.response)
