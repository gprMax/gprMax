from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_WAVE_TYPE = "gaussiandot"
SPECTRAL_AMPLITUDE_THRESHOLD = 1e-3
SPECTRUM_DB_FLOOR = -100.0
RAW_SPECTRUM_DYNAMIC_RANGE_DB = 120.0


def _integrate(values: np.ndarray, x: np.ndarray, axis: int = -1):
    """Integrate with both current and pre-2.0 NumPy releases."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


def _receiver_number(name: str) -> int:
    return int(name.removeprefix("rx"))


def _source_waveform(times: np.ndarray, frequency: float) -> np.ndarray:
    if SOURCE_WAVE_TYPE != "gaussiandot":
        raise ValueError(f"Unsupported comparison waveform {SOURCE_WAVE_TYPE!r}")
    chi = 1.0 / frequency
    zeta = 2.0 * np.pi**2 * frequency**2
    delay = times - chi
    return -2.0 * zeta * delay * np.exp(-zeta * delay**2)


def _read_line_plane(
    handle: h5py.File,
    x_position: float,
    electric_component: str,
    magnetic_component: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = []
    for name in sorted(handle["rxs"], key=_receiver_number):
        receiver = handle[f"rxs/{name}"]
        position = np.asarray(receiver.attrs["Position"], dtype=np.float64)
        if np.isclose(position[0], x_position):
            records.append(
                (
                    float(position[1]),
                    np.asarray(receiver[electric_component], dtype=np.float64),
                    np.asarray(receiver[magnetic_component], dtype=np.float64),
                )
            )
    if not records:
        raise ValueError(f"No receiver plane found at x={x_position:g} m")
    records.sort(key=lambda item: item[0])
    y = np.asarray([item[0] for item in records])
    electric = np.stack([item[1] for item in records], axis=1)
    magnetic = np.stack([item[2] for item in records], axis=1)
    return y, electric, magnetic


def _read_area_plane(
    handle: h5py.File,
    x_position: float,
) -> tuple[np.ndarray, ...]:
    records = []
    field_names = ("Ey", "Ez", "Hy", "Hz")
    for name in sorted(handle["rxs"], key=_receiver_number):
        receiver = handle[f"rxs/{name}"]
        position = np.asarray(receiver.attrs["Position"], dtype=np.float64)
        if np.isclose(position[0], x_position):
            records.append(
                (
                    float(position[1]),
                    float(position[2]),
                    *(
                        np.asarray(receiver[field], dtype=np.float64)
                        for field in field_names
                    ),
                )
            )
    if not records:
        raise ValueError(f"No receiver plane found at x={x_position:g} m")

    y = np.asarray(sorted({record[0] for record in records}))
    z = np.asarray(sorted({record[1] for record in records}))
    y_index = {value: index for index, value in enumerate(y)}
    z_index = {value: index for index, value in enumerate(z)}
    sample_count = records[0][2].size
    fields = {
        name: np.empty((sample_count, y.size, z.size), dtype=np.float64)
        for name in field_names
    }
    filled = np.zeros((y.size, z.size), dtype=bool)
    for record in records:
        yi = y_index[record[0]]
        zi = z_index[record[1]]
        filled[yi, zi] = True
        for name, values in zip(field_names, record[2:]):
            fields[name][:, yi, zi] = values
    if not np.all(filled):
        raise ValueError(f"Receiver plane at x={x_position:g} m is not rectangular")
    return y, z, *(fields[name] for name in field_names)


def _time_aligned_line_power(
    y: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    cross_sign: float,
) -> np.ndarray:
    """Estimate x-directed power with H interpolated to the E sample times."""
    magnetic_at_e = _align_h_to_e_times(magnetic)
    poynting_x = cross_sign * electric * magnetic_at_e
    return _integrate(poynting_x, x=y, axis=1)


def _align_h_to_e_times(magnetic: np.ndarray) -> np.ndarray:
    """Interpolate staggered H samples onto the electric-field time grid."""
    magnetic_at_e = magnetic.copy()
    magnetic_at_e[:-1] = 0.5 * (magnetic[:-1] + magnetic[1:])
    return magnetic_at_e


def _frequency_domain_line_power(
    y: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    cross_sign: float,
    dt: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x-directed spectral power from the complex FFT-domain fields."""
    electric_spectrum = np.fft.rfft(electric, n=nfft, axis=0)
    magnetic_spectrum = np.fft.rfft(
        _align_h_to_e_times(magnetic), n=nfft, axis=0
    )
    poynting_x = (
        0.5
        * cross_sign
        * electric_spectrum
        * np.conj(magnetic_spectrum)
    )
    power = np.real(_integrate(poynting_x, x=y, axis=1))
    frequencies = np.fft.rfftfreq(nfft, d=dt)
    return frequencies, power


def _integrate_area(
    values: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    return _integrate(_integrate(values, x=z, axis=2), x=y, axis=1)


def _time_aligned_area_power(
    y: np.ndarray,
    z: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
) -> np.ndarray:
    """Integrate Sx = Ey*Hz - Ez*Hy over a 3D receiver plane."""
    poynting_x = (
        ey * _align_h_to_e_times(hz)
        - ez * _align_h_to_e_times(hy)
    )
    return _integrate_area(poynting_x, y, z)


def _frequency_domain_area_power(
    y: np.ndarray,
    z: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    dt: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3D x-directed spectral power integrated over y and z."""
    ey_spectrum = np.fft.rfft(ey, n=nfft, axis=0)
    ez_spectrum = np.fft.rfft(ez, n=nfft, axis=0)
    hy_spectrum = np.fft.rfft(
        _align_h_to_e_times(hy), n=nfft, axis=0
    )
    hz_spectrum = np.fft.rfft(
        _align_h_to_e_times(hz), n=nfft, axis=0
    )
    poynting_x = 0.5 * (
        ey_spectrum * np.conj(hz_spectrum)
        - ez_spectrum * np.conj(hy_spectrum)
    )
    frequencies = np.fft.rfftfreq(nfft, d=dt)
    return frequencies, np.real(_integrate_area(poynting_x, y, z))


def _normalised_spectrum_db(
    measured_power: np.ndarray,
    source_power: np.ndarray,
    source_support: np.ndarray,
) -> np.ndarray:
    """Normalize by source spectral power and convert supported bins to dB."""
    ratio = np.full_like(source_power, np.nan, dtype=np.float64)
    ratio[source_support] = (
        np.maximum(measured_power[source_support], 0.0)
        / source_power[source_support]
    )
    floor = 10 ** (SPECTRUM_DB_FLOOR / 10)
    ratio[source_support] = np.maximum(ratio[source_support], floor)
    return 10 * np.log10(ratio)


def _raw_spectrum_db(
    power: np.ndarray,
    source_peak: float,
) -> np.ndarray:
    """Convert raw FFT power to dB using a common source-relative display floor."""
    floor = source_peak * 10 ** (-RAW_SPECTRUM_DYNAMIC_RANGE_DB / 10)
    return 10 * np.log10(np.maximum(power, floor))


def _build_case_result(
    dt: float,
    source_frequency: float,
    upstream_power: np.ndarray,
    downstream_power: np.ndarray,
    frequencies: np.ndarray,
    upstream_spectrum: np.ndarray,
    downstream_spectrum: np.ndarray,
    nfft: int,
) -> dict[str, np.ndarray | float]:
    times = dt * np.arange(upstream_power.size)
    waveform = _source_waveform(times, source_frequency)
    waveform_spectrum = np.fft.rfft(waveform, n=nfft)
    source_power_spectrum = np.abs(waveform_spectrum) ** 2
    source_support = source_power_spectrum >= (
        SPECTRAL_AMPLITUDE_THRESHOLD**2 * np.max(source_power_spectrum)
    )
    forward_power_spectrum = np.maximum(downstream_spectrum, 0.0)
    backward_power_spectrum = np.maximum(-upstream_spectrum, 0.0)

    # The modal solvers use 0.5*integral(E x H*) = 1 W (or W/m in 2D).
    # The corresponding instantaneous real-profile reference is 2*g(t)^2;
    # the frequency-domain reference is |G(f)|^2.
    source_power = 2.0 * waveform**2
    source_peak = float(np.max(source_power))
    source_energy = float(_integrate(source_power, x=times))
    forward = np.maximum(downstream_power, 0.0)
    backward = np.maximum(-upstream_power, 0.0)
    return {
        "times": times,
        "forward": forward / source_peak,
        "backward": backward / source_peak,
        "forward_energy": float(_integrate(forward, x=times) / source_energy),
        "backward_energy": float(_integrate(backward, x=times) / source_energy),
        "frequencies": frequencies,
        "source_support": source_support,
        "source_power_spectrum": source_power_spectrum,
        "forward_power_spectrum": forward_power_spectrum,
        "backward_power_spectrum": backward_power_spectrum,
        "forward_spectrum_db": _normalised_spectrum_db(
            forward_power_spectrum,
            source_power_spectrum,
            source_support,
        ),
        "backward_spectrum_db": _normalised_spectrum_db(
            backward_power_spectrum,
            source_power_spectrum,
            source_support,
        ),
    }


def load_2d_case(
    path: Path,
    source_frequency: float,
    upstream_x: float,
    downstream_x: float,
    electric_component: str,
    magnetic_component: str,
    cross_sign: float,
) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs["dt"])
        upstream = _read_line_plane(
            handle,
            upstream_x,
            electric_component,
            magnetic_component,
        )
        downstream = _read_line_plane(
            handle,
            downstream_x,
            electric_component,
            magnetic_component,
        )
    upstream_power = _time_aligned_line_power(
        *upstream, cross_sign=cross_sign
    )
    downstream_power = _time_aligned_line_power(
        *downstream, cross_sign=cross_sign
    )
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * upstream_power.size))))
    frequencies, upstream_spectrum = _frequency_domain_line_power(
        *upstream,
        cross_sign=cross_sign,
        dt=dt,
        nfft=nfft,
    )
    downstream_frequencies, downstream_spectrum = (
        _frequency_domain_line_power(
            *downstream,
            cross_sign=cross_sign,
            dt=dt,
            nfft=nfft,
        )
    )
    if not np.array_equal(frequencies, downstream_frequencies):
        raise RuntimeError("Receiver planes produced inconsistent frequency grids")
    return _build_case_result(
        dt,
        source_frequency,
        upstream_power,
        downstream_power,
        frequencies,
        upstream_spectrum,
        downstream_spectrum,
        nfft,
    )


def load_3d_case(
    path: Path,
    source_frequency: float,
    upstream_x: float,
    downstream_x: float,
) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs["dt"])
        upstream = _read_area_plane(handle, upstream_x)
        downstream = _read_area_plane(handle, downstream_x)
    upstream_power = _time_aligned_area_power(*upstream)
    downstream_power = _time_aligned_area_power(*downstream)
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * upstream_power.size))))
    frequencies, upstream_spectrum = _frequency_domain_area_power(
        *upstream, dt=dt, nfft=nfft
    )
    downstream_frequencies, downstream_spectrum = (
        _frequency_domain_area_power(
            *downstream, dt=dt, nfft=nfft
        )
    )
    if not np.array_equal(frequencies, downstream_frequencies):
        raise RuntimeError("Receiver planes produced inconsistent frequency grids")
    return _build_case_result(
        dt,
        source_frequency,
        upstream_power,
        downstream_power,
        frequencies,
        upstream_spectrum,
        downstream_spectrum,
        nfft,
    )


def plot_comparison(
    root: Path,
    title: str,
    loader,
) -> tuple[Path, Path, Path]:
    cases = {
        "Single-frequency solve": root / "single_frequency" / "single_frequency.h5",
        "Seven-frequency solve": root / "seven_frequency" / "seven_frequency.h5",
    }
    missing = [str(path) for path in cases.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison output(s): " + ", ".join(missing))

    results = {label: loader(path) for label, path in cases.items()}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = ("tab:blue", "tab:orange")
    for color, (label, result) in zip(colors, results.items()):
        time_ns = np.asarray(result["times"]) * 1e9
        axes[0].plot(
            time_ns,
            result["forward"],
            color=color,
            label=f"{label}; energy={result['forward_energy']:.3f}",
        )
        axes[1].plot(
            time_ns,
            result["backward"],
            color=color,
            label=f"{label}; energy={result['backward_energy']:.3e}",
        )

    axes[0].set_ylabel("Forward power / source peak power")
    axes[1].set_ylabel("Backward power / source peak power")
    axes[1].set_xlabel("Time (ns)")
    axes[0].set_title(f"{title} eigenmode source comparison")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    time_output = root / "normalized_forward_backward_power.png"
    fig.savefig(time_output, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    supported_ranges = []
    for color, (label, result) in zip(colors, results.items()):
        frequencies_ghz = np.asarray(result["frequencies"]) * 1e-9
        support = np.asarray(result["source_support"], dtype=bool)
        supported_ranges.append(
            (frequencies_ghz[support][0], frequencies_ghz[support][-1])
        )
        axes[0].plot(
            frequencies_ghz[support],
            np.asarray(result["forward_spectrum_db"])[support],
            color=color,
            label=label,
        )
        axes[1].plot(
            frequencies_ghz[support],
            np.asarray(result["backward_spectrum_db"])[support],
            color=color,
            label=label,
        )

    axes[0].set_ylabel("Forward power / source spectrum (dB)")
    axes[1].set_ylabel("Backward power / source spectrum (dB)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[0].set_title(f"Source-normalized {title} power spectra")
    axes[1].set_xlim(
        min(bounds[0] for bounds in supported_ranges),
        max(bounds[1] for bounds in supported_ranges),
    )
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        axis.set_ylim(bottom=SPECTRUM_DB_FLOOR)
        axis.grid(True, alpha=0.3)
        axis.legend()

    spectrum_output = root / "normalized_forward_backward_power_spectrum_db.png"
    fig.savefig(spectrum_output, dpi=180)
    plt.close(fig)

    baseline = next(iter(results.values()))
    baseline_frequencies_ghz = np.asarray(baseline["frequencies"]) * 1e-9
    baseline_support = np.asarray(baseline["source_support"], dtype=bool)
    baseline_source_power = np.asarray(baseline["source_power_spectrum"])
    source_peak = float(np.max(baseline_source_power))
    source_db = _raw_spectrum_db(baseline_source_power, source_peak)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for axis in axes:
        axis.plot(
            baseline_frequencies_ghz[baseline_support],
            source_db[baseline_support],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Source power spectrum",
        )
    for color, (label, result) in zip(colors, results.items()):
        frequencies_ghz = np.asarray(result["frequencies"]) * 1e-9
        support = np.asarray(result["source_support"], dtype=bool)
        axes[0].plot(
            frequencies_ghz[support],
            _raw_spectrum_db(
                np.asarray(result["forward_power_spectrum"]),
                source_peak,
            )[support],
            color=color,
            label=label,
        )
        axes[1].plot(
            frequencies_ghz[support],
            _raw_spectrum_db(
                np.asarray(result["backward_power_spectrum"]),
                source_peak,
            )[support],
            color=color,
            label=label,
        )

    axes[0].set_ylabel("Forward spectral power (dB, raw FFT scale)")
    axes[1].set_ylabel("Backward spectral power (dB, raw FFT scale)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[0].set_title(f"Unnormalized {title} power spectra")
    axes[1].set_xlim(
        baseline_frequencies_ghz[baseline_support][0],
        baseline_frequencies_ghz[baseline_support][-1],
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    raw_spectrum_output = (
        root / "unnormalized_forward_backward_power_spectrum_db.png"
    )
    fig.savefig(raw_spectrum_output, dpi=180)
    plt.close(fig)
    return time_output, spectrum_output, raw_spectrum_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot normalized forward/backward power for the broadband eigenmode comparison."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="broadband_comparison directory; defaults to the script directory.",
    )
    root = parser.parse_args().root.resolve()
    comparisons = (
        (
            root,
            "2D TM dielectric slab",
            lambda path: load_2d_case(
                path,
                source_frequency=1e9,
                upstream_x=0.02,
                downstream_x=0.10,
                electric_component="Ez",
                magnetic_component="Hy",
                cross_sign=-1.0,
            ),
        ),
        (
            root / "2d_te",
            "2D TE dielectric slab",
            lambda path: load_2d_case(
                path,
                source_frequency=1e9,
                upstream_x=0.04,
                downstream_x=0.20,
                electric_component="Ey",
                magnetic_component="Hz",
                cross_sign=1.0,
            ),
        ),
        (
            root / "3d",
            "3D dielectric channel",
            lambda path: load_3d_case(
                path,
                source_frequency=0.5e9,
                upstream_x=0.10,
                downstream_x=0.50,
            ),
        ),
    )
    for comparison_root, title, loader in comparisons:
        for output in plot_comparison(comparison_root, title, loader):
            print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
