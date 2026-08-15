from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from toolboxes.FMCW.processing import (
    ChannelResponse,
    Chirp,
    interpolate_instrument_response,
    process_channel,
    process_incident_referenced_channel,
    reconstruct_fast_time,
    synthesize_deramped_sweep,
    write_fmcw_output,
)


def _channel(chirp, response):
    values = np.asarray(response, dtype=np.complex128)
    target = SimpleNamespace(
        response=values,
        source=SimpleNamespace(filename="target.h5", path="/srcs/src1"),
        receiver=SimpleNamespace(filename="target.h5", path="/rxs/rx1/Ez"),
        source_spectrum=np.ones(chirp.samples, dtype=np.complex128),
        receiver_spectrum=values,
    )
    return ChannelResponse(
        chirp=chirp,
        response=values,
        target=target,
        background=None,
        source_valid=np.ones(chirp.samples, dtype=bool),
        normalisation="test",
    )


def _write_signal_file(path, source_amplitude, receiver_samples, dt=2e-11):
    with h5py.File(path, "w") as output:
        output.attrs["dt"] = dt
        source = output.create_group("srcs/src1/excitation")
        source.attrs["SampleInterval"] = dt
        source.attrs["TimeSampleOffset"] = 0.0
        source.create_dataset(
            "samples",
            data=np.r_[source_amplitude, np.zeros(len(receiver_samples) - 1)],
        )
        receiver = output.create_group("rxs/rx1")
        field = receiver.create_dataset("Ez", data=receiver_samples)
        field.attrs["SampleInterval"] = dt
        field.attrs["TimeSampleOffset"] = 0.0


def test_chirp_uses_endpoint_exclusive_frequencies_and_physical_resolution():
    chirp = Chirp(150e6, 1.2e9, 0.1, 8)

    assert chirp.frequency[0] == 150e6
    assert chirp.frequency[-1] == 150e6 + 7 * (1.05e9 / 8)
    assert chirp.frequency_step == 1.05e9 / 8
    assert chirp.delay[1] == pytest.approx(1 / 1.05e9)
    assert chirp.delay[-1] < chirp.samples / chirp.bandwidth


def test_point_target_maps_to_correct_positive_beat_and_fast_time():
    chirp = Chirp(100e6, 900e6, 80e-6, 128)
    delay_index = 11
    delay = delay_index / chirp.bandwidth
    response = np.exp(-2j * np.pi * chirp.frequency * delay)
    channel = _channel(chirp, response)

    fast = reconstruct_fast_time(channel, window="rectangular")
    sweep = synthesize_deramped_sweep(channel)

    assert np.argmax(np.abs(fast.complex_envelope)) == delay_index
    assert fast.delay[delay_index] == pytest.approx(delay)
    assert np.argmax(np.abs(np.fft.fft(sweep.complex_signal))) == delay_index
    assert sweep.beat_frequency[delay_index] == pytest.approx(chirp.slope * delay)
    assert sweep.delay[delay_index] == pytest.approx(delay)


def test_deramped_and_direct_processing_are_algebraically_identical():
    chirp = Chirp(120e6, 1.02e9, 60e-6, 96)
    rng = np.random.default_rng(8)
    response = rng.normal(size=chirp.samples) + 1j * rng.normal(size=chirp.samples)
    channel = _channel(chirp, response)

    fast = reconstruct_fast_time(channel, window="rectangular")
    sweep = synthesize_deramped_sweep(channel)
    from_stretch_receiver = np.conj(np.fft.fft(sweep.complex_signal)) / chirp.samples

    assert_allclose(from_stretch_receiver, fast.complex_envelope, rtol=2e-14, atol=2e-14)


def test_residual_video_phase_is_exact_on_discrete_delay_grid():
    chirp = Chirp(1e9, 5e9, 2e-9, 128)
    delay_index = 17
    delay = delay_index / chirp.bandwidth
    response = np.exp(-2j * np.pi * chirp.frequency * delay)
    channel = _channel(chirp, response)

    plain = synthesize_deramped_sweep(channel)
    with_rvp = synthesize_deramped_sweep(channel, residual_video_phase="include")
    expected = np.exp(-1j * np.pi * chirp.slope * delay**2)

    assert_allclose(with_rvp.complex_signal / plain.complex_signal, expected, atol=2e-13)
    observed = np.conj(np.fft.fft(with_rvp.complex_signal)) / chirp.samples
    corrected = observed * np.exp(-1j * np.pi * chirp.slope * chirp.delay**2)
    reference = np.fft.ifft(response)
    assert_allclose(corrected, reference, rtol=3e-13, atol=3e-13)


def test_down_chirp_reverses_beat_sign_and_preserves_positive_delay_mapping():
    chirp = Chirp(200e6, 1.0e9, 50e-6, 128, direction="down")
    delay_index = 9
    delay = delay_index / chirp.bandwidth
    response = np.exp(-2j * np.pi * chirp.frequency * delay)
    channel = _channel(chirp, response)

    sweep = synthesize_deramped_sweep(channel)
    peak = int(np.argmax(np.abs(np.fft.fft(sweep.complex_signal))))

    assert sweep.beat_frequency[peak] < 0
    assert sweep.delay[peak] == pytest.approx(delay)


def test_target_and_background_are_normalised_before_subtraction(tmp_path):
    target_file = tmp_path / "target.h5"
    background_file = tmp_path / "background.h5"
    target_receiver = np.zeros(128)
    target_receiver[3] = 2.0
    target_receiver[9] = 0.5
    background_receiver = np.zeros(128)
    background_receiver[3] = 5.0
    _write_signal_file(target_file, 2.0, target_receiver)
    _write_signal_file(background_file, 5.0, background_receiver)
    chirp = Chirp(0.1e9, 1.1e9, 1e-3, 51)

    channel = process_channel(target_file, chirp, background_filename=background_file)

    expected = 0.25 * np.exp(-2j * np.pi * chirp.frequency * 9 * 2e-11)
    assert_allclose(channel.response, expected, rtol=3e-12, atol=3e-12)


def test_single_background_trace_broadcasts_over_merged_bscan(tmp_path):
    target_file = tmp_path / "target_bscan.h5"
    background_file = tmp_path / "background.h5"
    target_receiver = np.zeros((128, 2))
    target_receiver[3, :] = 2.0
    target_receiver[8, 0] = 0.5
    target_receiver[11, 1] = -0.25
    background_receiver = np.zeros(128)
    background_receiver[3] = 2.0
    _write_signal_file(target_file, 2.0, target_receiver)
    _write_signal_file(background_file, 2.0, background_receiver)
    chirp = Chirp(0.1e9, 1.1e9, 1e-3, 41)

    channel = process_channel(target_file, chirp, background_filename=background_file)

    assert channel.response.shape == (chirp.samples, 2)
    assert not np.allclose(channel.response[:, 0], channel.response[:, 1])


def test_incident_reference_normalises_plane_wave_style_total_field(tmp_path):
    total_file = tmp_path / "total.h5"
    incident_file = tmp_path / "incident.h5"
    incident = np.zeros(128)
    incident[3] = 2.0
    total = incident.copy()
    total[10] = -0.5
    _write_signal_file(total_file, 1.0, total)
    _write_signal_file(incident_file, 1.0, incident)
    chirp = Chirp(0.1e9, 1.1e9, 1e-3, 41)

    channel = process_incident_referenced_channel(total_file, incident_file, chirp)

    expected = -0.25 * np.exp(-2j * np.pi * chirp.frequency * 7 * 2e-11)
    assert channel.normalisation == "incident-reference"
    assert_allclose(channel.response, expected, rtol=3e-12, atol=3e-12)


def test_instrument_response_interpolates_unwrapped_phase():
    frequency = np.asarray((100e6, 200e6, 300e6))
    phase = np.deg2rad((170, 190, 210))
    response = np.asarray((1.0, 2.0, 3.0)) * np.exp(1j * phase)

    result = interpolate_instrument_response(
        frequency,
        response,
        np.asarray((150e6, 250e6)),
    )

    assert_allclose(np.abs(result), (1.5, 2.5))
    assert_allclose(np.unwrap(np.angle(result)), np.deg2rad((180, 200)), atol=2e-15)


def test_receiver_delay_response_is_applied_after_inverse_transform():
    chirp = Chirp(100e6, 500e6, 1e-3, 32)
    channel = _channel(chirp, np.ones(chirp.samples))
    delay_response = np.ones(chirp.samples)
    delay_response[0] = 0.25

    result = reconstruct_fast_time(
        channel,
        window="rectangular",
        receiver_delay_response=delay_response,
    )

    assert result.complex_envelope[0] == pytest.approx(0.25)
    assert_allclose(result.receiver_delay_response, delay_response)


def test_output_contains_fast_time_and_optional_deramped_products(tmp_path):
    chirp = Chirp(100e6, 500e6, 1e-3, 32)
    channel = _channel(chirp, np.ones(chirp.samples))
    fast = reconstruct_fast_time(channel, window="hann", propagation_velocity=2e8)
    deramped = synthesize_deramped_sweep(channel)
    output = tmp_path / "fmcw.h5"

    write_fmcw_output(output, channel, fast, deramped)

    with h5py.File(output, "r") as result:
        assert result.attrs["FrequencyEndpointIncluded"] == np.False_
        assert result["frequency"].shape == (32,)
        assert result["fast_time/range"].attrs["PropagationVelocity"] == 2e8
        assert result["fast_time"].attrs["Window"] == "hann"
        assert result["fast_time/receiver_delay_response"].shape == (32,)
        assert result["deramped_sweep/I"].shape == (32,)
