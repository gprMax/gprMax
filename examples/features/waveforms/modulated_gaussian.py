"""Plot and export the built-in, MATLAB-default gauspulse waveform.

Run from the repository root:
    python -m examples.features.waveforms.modulated_gaussian --output-dir waveform_preview
"""

import argparse
from pathlib import Path

import numpy as np

from gprMax.waveforms import Waveform


def waveform():
    """The gallery pulse: unit amplitude and a 1 GHz carrier."""
    w = Waveform()
    w.type, w.amp, w.freq = "gauspulse", 1.0, 1e9
    w.calculate_coefficients()
    return w


def sample_waveform():
    """Use the waveform gallery's 6 ns window and 1.926 ps sample interval."""
    dt = 1.926e-12
    times = np.arange(int(np.ceil(6e-9 / dt)) + 1) * dt
    w = waveform()
    return times, np.array([w.calculate_value(t, dt) for t in times])


def plot_and_export(output_dir, plot_path=None):
    """Save the plot and a two-column file accepted by #excitation_file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times, values = sample_waveform()
    sample_path = output_dir / "modulated_gaussian.txt"
    np.savetxt(
        sample_path,
        np.column_stack((times, values)),
        header="time modulated_gaussian",
        comments="",
    )

    dt = times[1] - times[0]
    # Zero padding smooths the displayed spectrum; it adds no information.
    nfft = 8 * (1 << (len(times) - 1).bit_length())
    frequencies = np.fft.rfftfreq(nfft, dt)
    magnitude = np.abs(np.fft.rfft(values, n=nfft))
    power_db = 20 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-8))
    w = waveform()
    envelope = np.exp(-w.zeta * (times - w.chi) ** 2)

    fig, (time_ax, frequency_ax) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    time_ax.plot(times * 1e9, values, color="firebrick", label="gauspulse (1 GHz)")
    time_ax.plot(times * 1e9, envelope, "--", color="0.45", label="Gaussian envelope")
    time_ax.plot(times * 1e9, -envelope, "--", color="0.45")
    time_ax.set(xlabel="Time (ns)", ylabel="Amplitude", xlim=(0, 6), ylim=(-1.1, 1.1))
    time_ax.legend(loc="upper right", fontsize=8)
    frequency_ax.plot(frequencies * 1e-9, power_db, color="firebrick")
    frequency_ax.axvline(1, color="0.45", linestyle="--", label="Carrier: 1 GHz")
    frequency_ax.plot([0.75, 1.25], [-6, -6], "o-", color="0.3", label="50% bandwidth at -6 dB")
    frequency_ax.set(
        xlabel="Frequency (GHz)",
        ylabel="Power relative to peak (dB)",
        xlim=(0, 2.5),
        ylim=(-80, 3),
    )
    frequency_ax.legend(loc="upper right", fontsize=8)
    for ax in (time_ax, frequency_ax):
        ax.grid(alpha=0.3)

    plot_path = Path(plot_path) if plot_path is not None else output_dir / "modulated_gaussian.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return sample_path, plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("waveform_preview"))
    parser.add_argument("--plot", type=Path, help="Override the plot filename, e.g. for the docs gallery")
    args = parser.parse_args()
    sample_path, plot_path = plot_and_export(args.output_dir, args.plot)
    print(f"Samples for #excitation_file: {sample_path}")
    print(f"Time trace and spectrum: {plot_path}")


if __name__ == "__main__":
    main()
