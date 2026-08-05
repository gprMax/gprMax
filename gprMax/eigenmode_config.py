"""Reusable frequency-band configuration for eigenmode ports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf


logger = logging.getLogger(__name__)


def _padded_sample_count(sample_count: int) -> int:
    return 1 << int(np.ceil(np.log2(max(2, 2 * sample_count))))


def sampled_waveform_spectrum(waveform, dt: float, sample_count: int):
    """Return the exact zero-padded spectrum used by eigenmode injection."""
    times = np.arange(sample_count, dtype=np.float64) * dt
    samples = np.asarray(
        [waveform.calculate_value(time, dt) for time in times],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(samples)):
        raise ValueError('Eigenmode excitation waveform contains non-finite samples.')
    padded_count = _padded_sample_count(sample_count)
    frequencies = np.fft.rfftfreq(padded_count, d=dt)
    spectrum = np.fft.rfft(samples, n=padded_count)
    return samples, frequencies, spectrum


class EigenmodeBandpassWaveform:
    """Finite band-pass pulse with Gaussian-smoothed asymmetric edges."""

    type = 'eigenmode_bandpass'

    def __init__(
        self,
        *,
        band_id: str,
        fmin: float,
        fmax: float,
        amplitude: float,
        dt: float,
        sample_count: int,
        spectral_threshold: float,
        transition: str | float = 'auto',
    ):
        if fmax <= fmin:
            raise ValueError(
                'Automatic eigenmode bandpass excitation requires fmax greater '
                'than fmin. Supply an explicit waveform for a single-frequency band.'
            )
        self.ID = f'{band_id}_auto_bandpass'
        self.amp = float(amplitude)
        self.freq = 0.5 * (fmin + fmax)
        self.dt = float(dt)
        self.sample_count = int(sample_count)
        self.spectral_threshold = float(spectral_threshold)

        padded_count = _padded_sample_count(self.sample_count)
        frequencies = np.fft.rfftfreq(padded_count, d=self.dt)
        if frequencies.size < 3:
            raise ValueError('The time window is too short to construct an eigenmode bandpass waveform.')
        frequency_step = frequencies[1]
        nyquist = frequencies[-1]
        if fmax >= nyquist:
            raise ValueError(
                f'Eigenmode band {band_id!r} reaches or exceeds Nyquist. Reduce '
                'fmax or the time step.'
            )

        passband = (frequencies >= fmin) & (frequencies <= fmax)
        if not np.any(passband):
            raise ValueError(
                f'The time window is too short to sample eigenmode band {band_id!r}. '
                'Increase the time window.'
            )

        bandwidth = fmax - fmin
        threshold_radius = np.sqrt(2.0 * np.log(1.0 / self.spectral_threshold))
        if transition == 'auto':
            time_window = self.sample_count * self.dt
            time_localising_sigma = threshold_radius / (np.pi * time_window)
            nominal_sigma = max(0.25 * bandwidth, time_localising_sigma)
            lower_sigma = min(
                nominal_sigma,
                0.9 * fmin / threshold_radius,
            )
            upper_sigma = min(
                nominal_sigma,
                0.9 * (nyquist - fmax) / threshold_radius,
            )
        else:
            transition_width = float(transition)
            if not np.isfinite(transition_width) or transition_width <= 0:
                raise ValueError('Eigenmode band transition width must be positive or auto.')
            lower_sigma = upper_sigma = transition_width / threshold_radius
        if lower_sigma <= 0 or upper_sigma <= 0:
            raise ValueError(
                f'Eigenmode band {band_id!r} has insufficient spectral guard space.'
            )

        self.lower_stop = fmin - threshold_radius * lower_sigma
        self.upper_stop = fmax + threshold_radius * upper_sigma
        if self.lower_stop <= 0 or self.upper_stop >= nyquist:
            raise ValueError(
                f'Eigenmode band {band_id!r} is too close to DC or Nyquist for '
                'the requested spectral coverage. Move the band away from the '
                'limit or reduce the time step.'
            )

        lower_edge = 0.5 * (
            1.0 + erf((frequencies - fmin) / (np.sqrt(2.0) * lower_sigma))
        )
        upper_edge = 0.5 * (
            1.0 - erf((frequencies - fmax) / (np.sqrt(2.0) * upper_sigma))
        )
        target_magnitude = lower_edge * upper_edge

        # The zero-phase pulse is periodic and centred at index zero. Find the
        # part of its analytic envelope above the configured threshold, then
        # apply the earliest causal delay that keeps that complete support in
        # the sampled record. Centring the pulse in the full simulation window
        # unnecessarily consumes half of the propagation and ring-down time.
        analytic_spectrum = np.zeros(padded_count, dtype=np.complex128)
        analytic_spectrum[0] = target_magnitude[0]
        analytic_spectrum[1 : target_magnitude.size - 1] = (
            2.0 * target_magnitude[1:-1]
        )
        analytic_spectrum[padded_count // 2] = target_magnitude[-1]
        zero_phase_envelope = np.abs(np.fft.ifft(analytic_spectrum))
        envelope_peak = float(np.max(zero_phase_envelope, initial=0.0))
        time_significant = np.flatnonzero(
            zero_phase_envelope >= self.spectral_threshold * envelope_peak
        )
        circular_offsets = np.minimum(
            time_significant,
            padded_count - time_significant,
        )
        half_width_samples = int(np.max(circular_offsets, initial=0))
        if 2 * half_width_samples + 1 > self.sample_count:
            raise ValueError(
                f'The time window is too short to contain the automatic waveform '
                f'for eigenmode band {band_id!r}. Increase the time window or widen '
                'the frequency band.'
            )
        delay_samples = half_width_samples
        if 2 * half_width_samples + 2 <= self.sample_count:
            delay_samples += 1
        self.chi = delay_samples * self.dt
        delay = self.chi
        target_spectrum = target_magnitude * np.exp(-2j * np.pi * frequencies * delay)
        samples = np.fft.irfft(target_spectrum, n=padded_count)[: self.sample_count]
        samples -= np.mean(samples)
        nyquist_basis = (-1.0) ** np.arange(self.sample_count)
        samples -= nyquist_basis * np.dot(samples, nyquist_basis) / self.sample_count
        peak = float(np.max(np.abs(samples), initial=0.0))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f'Could not construct automatic waveform for eigenmode band {band_id!r}.')
        self.samples = np.asarray(samples * (self.amp / peak), dtype=np.float64)

        _, analysed_frequencies, analysed_spectrum = sampled_waveform_spectrum(
            self,
            self.dt,
            self.sample_count,
        )
        magnitude = np.abs(analysed_spectrum)
        analysed_peak = float(np.max(magnitude, initial=0.0))
        significant = magnitude >= self.spectral_threshold * analysed_peak
        indices = np.flatnonzero(significant)
        if not indices.size:
            raise ValueError(f'Automatic waveform for eigenmode band {band_id!r} has no significant spectrum.')
        if significant[0] or significant[-1]:
            raise ValueError(
                f'Automatic waveform for eigenmode band {band_id!r} cannot decay '
                'below the spectral threshold before DC or Nyquist. Increase the '
                'time window or move the band away from the frequency limit.'
            )
        self.significant_low = float(analysed_frequencies[indices[0]])
        self.significant_high = float(analysed_frequencies[indices[-1]])
        self.transition_widths = (
            float(fmin - self.lower_stop),
            float(self.upper_stop - fmax),
        )
        logger.info(
            f'Prepared automatic eigenmode bandpass {self.ID!r}: requested passband '
            f'{fmin:g} to {fmax:g} Hz, estimated Gaussian-edge threshold frequencies '
            f'{self.lower_stop:g} and {self.upper_stop:g} Hz, significant sampled '
            f'spectrum {self.significant_low:g} to {self.significant_high:g} Hz, '
            f'and pulse centre {self.chi:g} s.'
        )

    def calculate_value(self, time, dt):
        if time < 0:
            return 0.0
        index = int(round(time / self.dt))
        if index < 0 or index >= self.samples.size:
            return 0.0
        return float(self.samples[index])

    def calculate_coefficients(self):
        return None


def _format_anchor_suggestion(low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return ''
    anchors = np.linspace(low, high, 5)
    return ', '.join(f'{value:.12g}' for value in anchors)


def automatic_anchor_frequencies(
    support_low: float,
    support_high: float,
    fmin: float,
    fmax: float,
) -> tuple[float, ...]:
    """Choose deterministic log-spaced anchors plus passband landmarks."""
    if support_low > 0 and support_high == support_low:
        return (float(support_low),)
    if support_low <= 0 or support_high <= support_low:
        raise ValueError('Automatic eigenmode anchor support must satisfy 0 < low < high.')
    ratio = support_high / support_low
    intervals = max(2, int(np.ceil(np.log(ratio) / np.log(1.5))))
    intervals = min(intervals, 8)
    generated = np.geomspace(support_low, support_high, intervals + 1)
    landmarks = np.asarray(
        (support_low, fmin, 0.5 * (fmin + fmax), fmax, support_high),
        dtype=np.float64,
    )
    combined = np.unique(np.concatenate((generated, landmarks)))
    return tuple(float(value) for value in combined)


@dataclass
class EigenmodeBandSpec:
    id: str
    fmin: float
    fmax: float
    points: int
    transition: str | float = 'auto'
    spectral_threshold: float = 1e-3
    representative_frequency: float | None = None
    significant_range: tuple[float, float] | None = None

    def resolve_spectrum(self, grid, waveform, *, generated_waveform: bool):
        _, frequencies, spectrum = sampled_waveform_spectrum(
            waveform,
            grid.dt,
            int(grid.iterations),
        )
        magnitude = np.abs(spectrum)
        peak = float(np.max(magnitude, initial=0.0))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f'Eigenmode band {self.id!r} excitation has no finite spectral energy.')
        significant = magnitude >= self.spectral_threshold * peak
        indices = np.flatnonzero(significant)
        if not indices.size:
            raise ValueError(f'Eigenmode band {self.id!r} excitation has no significant spectrum.')
        significant_low = float(frequencies[indices[0]])
        significant_high = float(frequencies[indices[-1]])
        self.significant_range = (significant_low, significant_high)
        positive = frequencies > 0
        positive_indices = np.flatnonzero(positive)
        peak_positive = positive_indices[int(np.argmax(magnitude[positive]))]
        self.representative_frequency = float(frequencies[peak_positive])

        single_frequency = self.points == 1 and self.fmin == self.fmax
        if single_frequency:
            frequency_step = frequencies[1] if frequencies.size > 1 else np.inf
            declared_frequency = getattr(waveform, 'freq', None)
            comparison_frequency = (
                float(declared_frequency)
                if declared_frequency is not None and np.isfinite(declared_frequency)
                else self.representative_frequency
            )
            if abs(comparison_frequency - self.fmin) > frequency_step:
                raise ValueError(
                    f'Waveform frequency {comparison_frequency:g} Hz does not match '
                    f'single-frequency eigenmode band {self.fmin:g} Hz. Use a '
                    'matching continuous waveform or request a finite frequency band '
                    "with waveform='auto'."
                )
            self.representative_frequency = self.fmin
            self.significant_range = (self.fmin, self.fmax)
            return

        if significant[0] or significant[-1]:
            raise ValueError(
                f'Eigenmode band {self.id!r} excitation has significant DC or '
                'Nyquist content. Use the automatic eigenmode bandpass waveform.'
            )

        power = magnitude**2
        outside_requested = (frequencies < self.fmin) | (frequencies > self.fmax)
        outside_power = float(np.sum(power[outside_requested]) / np.sum(power))
        if not generated_waveform and outside_power > 0.01:
            raise ValueError(
                f'Waveform spectrum is unsuitable for eigenmode band {self.id!r}: '
                f'{100 * outside_power:.3f}% of its sampled spectral power lies outside '
                f'the requested {self.fmin:g} to {self.fmax:g} Hz band, and significant '
                f'bins span {significant_low:g} to {significant_high:g} Hz. Use '
                "EigenmodeExcitation(..., waveform='auto') to generate the reusable "
                'bandpass excitation.'
            )

@dataclass
class EigenmodePortSpec:
    port: int
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]
    normal: str
    direction: str
    normal_axis: int
    transverse_axes: tuple[int, int]
    invariant_axis: int | None
    modes: tuple[int, ...]
    anchors: str | tuple[float, ...]
    plot_fields: bool | None
    resolved_anchors: tuple[float, ...] = field(default_factory=tuple)

    @property
    def anchor_policy(self) -> str:
        return 'auto' if self.anchors == 'auto' else 'explicit'

    def resolve_anchors(self, band: EigenmodeBandSpec, *, is_source: bool):
        if band.significant_range is None:
            raise ValueError('Eigenmode excitation spectrum must be resolved before port anchors.')
        required_low = min(band.fmin, band.significant_range[0])
        required_high = max(band.fmax, band.significant_range[1])

        if self.anchors == 'auto':
            self.resolved_anchors = automatic_anchor_frequencies(
                required_low,
                required_high,
                band.fmin,
                band.fmax,
            )
        else:
            explicit = tuple(float(value) for value in self.anchors)
            if len(explicit) > 1 and (
                required_low < explicit[0] or required_high > explicit[-1]
            ):
                suggestion = _format_anchor_suggestion(required_low, required_high)
                raise ValueError(
                    f'Explicit eigenmode anchors for port {self.port} span '
                    f'{explicit[0]:g} to {explicit[-1]:g} Hz, but required modal '
                    f'coverage spans {required_low:g} to {required_high:g} Hz. '
                    f'Suggested coverage anchors: {suggestion}. Alternatively use '
                    'anchors=\'auto\'. A single explicit anchor is also accepted as an '
                    'intentional constant modal basis across the band.'
                )
            self.resolved_anchors = explicit
        logger.info(
            f'Resolved eigenmode port {self.port} modal anchors: '
            + ', '.join(f'{value:g}' for value in self.resolved_anchors)
            + ' Hz.'
        )
