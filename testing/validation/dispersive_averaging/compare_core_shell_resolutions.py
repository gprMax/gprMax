"""Compare baseline and refined Debye--Lorentz core-shell sphere results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gprmax-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "results" / "core_shell_fdtd"
DEFAULT_REFINED = DEFAULT_BASELINE / "refined"


def _load(directory):
    sampled = np.genfromtxt(directory / "core_shell_backscatter.csv", delimiter=",", names=True)
    dense = np.genfromtxt(directory / "core_shell_analytical_dense.csv", delimiter=",", names=True)
    return sampled, dense


def _metrics(simulated, analytical):
    error_db = 10 * np.log10(simulated / analytical)
    return {
        "frequency_samples": int(simulated.size),
        "relative_l2_error_percent": float(
            100 * np.linalg.norm(simulated - analytical) / np.linalg.norm(analytical)
        ),
        "rms_error_db": float(np.sqrt(np.mean(error_db**2))),
        "median_absolute_error_db": float(np.median(np.abs(error_db))),
        "maximum_absolute_error_db": float(np.max(np.abs(error_db))),
    }


def _common_indices(baseline_frequency, refined_frequency):
    refined_indices = np.searchsorted(refined_frequency, baseline_frequency)
    if np.any(refined_indices >= refined_frequency.size) or not np.allclose(
        refined_frequency[refined_indices], baseline_frequency, rtol=0, atol=1.0
    ):
        raise ValueError("The refined spectrum does not contain every baseline frequency")
    return refined_indices


def compare(baseline_dir, refined_dir, output_dir):
    baseline, _ = _load(baseline_dir)
    refined, dense = _load(refined_dir)
    common = _common_indices(baseline["frequency_hz"], refined["frequency_hz"])
    summary = {
        "common_frequency_samples": int(common.size),
        "refined_frequency_samples": int(refined.size),
        "modes": {},
    }
    for mode in ("averaged", "staircased"):
        summary["modes"][mode] = {
            "baseline_common_frequencies": _metrics(
                baseline[f"{mode}_rcs_m2"], baseline["analytical_rcs_m2"]
            ),
            "refined_common_frequencies": _metrics(
                refined[f"{mode}_rcs_m2"][common],
                refined["analytical_rcs_m2"][common],
            ),
            "refined_all_frequencies": _metrics(
                refined[f"{mode}_rcs_m2"], refined["analytical_rcs_m2"]
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolution_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    area = np.pi * 0.1**2
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.0, 1)},
    )
    colours = {"baseline": "black", "refined": "black"}
    for column, mode in enumerate(("averaged", "staircased")):
        rcs_axis = axes[0, column]
        error_axis = axes[1, column]
        rcs_axis.semilogy(
            dense["frequency_hz"] / 1e9,
            dense["analytical_rcs_m2"] / area,
            color="black",
            linewidth=2,
            label="Aden--Kerker analytical",
        )
        for label, table, marker in (
            ("4.0-mm baseline", baseline, "s"),
            ("2.5-mm refined", refined, "o"),
        ):
            key = "baseline" if "baseline" in label else "refined"
            rcs_axis.semilogy(
                table["frequency_hz"] / 1e9,
                table[f"{mode}_rcs_m2"] / area,
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
                markersize=3.4,
                color=colours[key],
                label=label,
            )
            error_axis.plot(
                table["frequency_hz"] / 1e9,
                np.abs(10 * np.log10(table[f"{mode}_rcs_m2"] / table["analytical_rcs_m2"])),
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
                markersize=3.4,
                color=colours[key],
                label=label,
            )
        title = "General pole average" if mode == "averaged" else "Staircased"
        rcs_axis.set(
            title=title,
            ylabel=r"Normalised backscatter, $\sigma_b/(\pi a^2)$",
        )
        error_axis.set(
            xlabel="Frequency [GHz]",
            ylabel="Pointwise absolute error [dB]",
        )
        for axis in (rcs_axis, error_axis):
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize="small")
    figure.savefig(output_dir / "core_shell_resolution_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--refined-dir", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REFINED)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(args.baseline_dir, args.refined_dir, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
