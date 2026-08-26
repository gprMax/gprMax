"""Plot PEC-versus-copper TE10 effective index, loss, and wall field."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLE_DIR / "rectangular_waveguide_eigenmode_comparison.png"
FMIN = 130e9
FMAX = 150e9
C0 = 299_792_458.0


def read_neff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read valid mode-1 FDFD anchor effective indices in the plotted band."""

    with h5py.File(path, "r") as output:
        group = output["eigenmode_ports/port1"]
        frequencies = np.asarray(
            group.attrs["CandidateAnchorFrequencies"],
            dtype=np.float64,
        )
        neff = np.asarray(group["anchor_complex_neff"], dtype=np.complex128)[:, 0]
        valid = np.asarray(group["anchor_mode_valid"], dtype=bool)[:, 0]
        reference_valid = np.asarray(
            group["anchor_mode_reference_valid"],
            dtype=bool,
        )[:, 0]
    selected = valid & reference_valid & (frequencies >= FMIN) & (frequencies <= FMAX)
    if np.count_nonzero(selected) < 2:
        raise RuntimeError(f"{path} has fewer than two valid in-band TE10 anchors")
    order = np.argsort(frequencies[selected])
    return frequencies[selected][order], neff[selected][order]


def read_receiver(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one named receiver's Ez samples and their stagger-aware times."""

    with h5py.File(path, "r") as output:
        receivers = output.get("rxs")
        if receivers is None:
            raise KeyError(f"{path} contains no receiver outputs")
        for group in receivers.values():
            stored_name = group.attrs.get("Name", "")
            if isinstance(stored_name, bytes):
                stored_name = stored_name.decode("utf-8")
            if stored_name != name:
                continue
            dataset = group["Ez"]
            values = np.asarray(dataset, dtype=np.float64)
            interval = float(dataset.attrs["SampleInterval"])
            offset = float(dataset.attrs["TimeSampleOffset"])
            time = offset + interval * np.arange(values.size, dtype=np.float64)
            return time, values
    raise KeyError(f"{path} has no receiver named {name!r}")


def wall_trace(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return wall Ez normalized by peak centre Ez and its peak ratio."""

    centre_time, centre = read_receiver(path, "aperture_centre")
    wall_time, wall = read_receiver(path, "lower_sidewall_tangential")
    if not np.array_equal(centre_time, wall_time):
        raise RuntimeError(f"receiver sample times differ in {path}")
    centre_peak = float(np.max(np.abs(centre), initial=0.0))
    if not np.isfinite(centre_peak) or centre_peak == 0:
        raise RuntimeError(f"{path} has no finite non-zero aperture-centre Ez")
    normalized = wall / centre_peak
    ratio = float(np.max(np.abs(normalized), initial=0.0))
    return wall_time, normalized, ratio


def attenuation_db_per_m(frequency: np.ndarray, neff: np.ndarray) -> np.ndarray:
    """Convert the passive e^(-j beta x) neff convention to positive dB/m."""

    alpha = -(2 * np.pi * frequency / C0) * np.imag(neff)
    return 20 / np.log(10) * alpha


def plot_comparison(pec_path: Path, copper_path: Path, destination: Path) -> None:
    """Write the three-panel modal and time-domain comparison."""

    pec_frequency, pec_neff = read_neff(pec_path)
    copper_frequency, copper_neff = read_neff(copper_path)
    np.testing.assert_allclose(pec_frequency, copper_frequency, rtol=0, atol=1e-6)

    pec_time, pec_wall, pec_ratio = wall_trace(pec_path)
    copper_time, copper_wall, copper_ratio = wall_trace(copper_path)
    if copper_ratio == 0:
        raise RuntimeError("copper wall-tangential Ez is unexpectedly zero")

    display_ratio = max(copper_ratio, pec_ratio, np.finfo(np.float64).tiny)
    display_exponent = max(0, int(np.floor(-np.log10(display_ratio))))
    display_scale = 10.0**display_exponent

    figure = Figure(
        figsize=(8.2, 10.0),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)
    axes[0].plot(
        pec_frequency * 1e-9,
        np.real(pec_neff),
        "o-",
        linewidth=2,
        label="PEC",
    )
    axes[0].plot(
        copper_frequency * 1e-9,
        np.real(copper_neff),
        "s-",
        linewidth=2,
        label="Copper surface impedance",
    )
    axes[0].set_ylabel(r"$\mathrm{Re}(n_\mathrm{eff})$")
    axes[0].set_title("FDFD TE10 phase constant")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        pec_frequency * 1e-9,
        attenuation_db_per_m(pec_frequency, pec_neff),
        "o-",
        linewidth=2,
        label="PEC",
    )
    axes[1].plot(
        copper_frequency * 1e-9,
        attenuation_db_per_m(copper_frequency, copper_neff),
        "s-",
        linewidth=2,
        label="Copper surface impedance",
    )
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("Modal attenuation (dB/m)")
    axes[1].set_title(r"Loss from $-k_0\,\mathrm{Im}(n_\mathrm{eff})$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        pec_time * 1e12,
        display_scale * pec_wall,
        linewidth=1.5,
        label=f"PEC (peak ratio {pec_ratio:.3g})",
    )
    axes[2].plot(
        copper_time * 1e12,
        display_scale * copper_wall,
        linewidth=1.5,
        label=f"Copper (peak ratio {copper_ratio:.3g})",
    )
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(rf"$10^{{{display_exponent}}} E_{{z,\,wall}}/\max|E_{{z,\,centre}}|$")
    axes[2].set_title("Tangential Ez at the lower side wall")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle("Rectangular TE10 waveguide: PEC versus copper walls")
    figure.savefig(destination, dpi=180)
    figure.clear()

    centre_index = int(np.argmin(np.abs(copper_frequency - 140e9)))
    print(f"At {copper_frequency[centre_index] * 1e-9:g} GHz:")
    print(f"  PEC neff:    {pec_neff[centre_index]:.9g}")
    print(f"  copper neff: {copper_neff[centre_index]:.9g}")
    print(
        "  copper attenuation: "
        f"{attenuation_db_per_m(copper_frequency, copper_neff)[centre_index]:.6g} dB/m"
    )
    print(f"Peak |Ez_wall|/|Ez_centre|, PEC:    {pec_ratio:.6g}")
    print(f"Peak |Ez_wall|/|Ez_centre|, copper: {copper_ratio:.6g}")
    print(f"Wrote {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pec",
        type=Path,
        default=EXAMPLE_DIR / "pec_rectangular_waveguide.h5",
    )
    parser.add_argument(
        "--copper",
        type=Path,
        default=EXAMPLE_DIR / "copper_rectangular_waveguide.h5",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_comparison(args.pec, args.copper, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
