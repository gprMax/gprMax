"""Plot the gprMax, CST FIT, and CST FEM patch-antenna S11 results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
GPRMAX_S11 = HERE / "patch_antenna_sparameters.csv"
CST_FIT_S11 = HERE / "patch_s11_fit_cst.s1p"
CST_FEM_S11 = HERE / "patch_s11_fem_cst.s1p"
SUMMARY = HERE / "patch_sparameter_summary.json"
SOLVER_STYLES = {
    "gprMax": {
        "color": "#0072b2",
        "linestyle": "-",
        "linewidth": 2.4,
        "zorder": 3,
    },
    "CST FIT": {
        "color": "#e69f00",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.4,
        "zorder": 5,
    },
    "CST FEM": {
        "color": "#cc79a7",
        "linestyle": (0, (2, 2)),
        "linewidth": 2.4,
        "zorder": 6,
    },
}
PLOT_ORDER = ("gprMax", "CST FIT", "CST FEM")


def read_gprmax_s11() -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with GPRMAX_S11.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                int(row["destination_port"]) == 1
                and int(row["destination_mode"]) == 1
                and bool(int(row["power_wave_valid"]))
            ):
                rows.append(
                    (float(row["frequency_hz"]), float(row["S_magnitude_db"]))
                )
    if not rows:
        raise ValueError(f"No valid port-1 mode-1 S11 samples in {GPRMAX_S11}")
    data = np.asarray(sorted(rows), dtype=np.float64)
    return data[:, 0], data[:, 1]


def read_touchstone_s11(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "#")):
            continue
        frequency_ghz, magnitude, _phase_deg = map(float, stripped.split()[:3])
        rows.append((frequency_ghz * 1e9, 20.0 * np.log10(magnitude)))
    if not rows:
        raise ValueError(f"No S11 samples in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1]


def plot_s11(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[float, float]]:
    minima: dict[str, tuple[float, float]] = {}
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for label in PLOT_ORDER:
        frequency, s11_db = series[label]
        minimum_index = int(np.nanargmin(s11_db))
        minimum_frequency = float(frequency[minimum_index])
        minimum_s11 = float(s11_db[minimum_index])
        minima[label] = minimum_frequency, minimum_s11
        (line,) = axis.plot(
            frequency * 1e-9,
            s11_db,
            label=label,
            **SOLVER_STYLES[label],
        )
        axis.plot(
            minimum_frequency * 1e-9,
            minimum_s11,
            "o",
            color=line.get_color(),
            label=(
                f"{label} min: {minimum_frequency * 1e-9:.3f} GHz, "
                f"{minimum_s11:.1f} dB"
            ),
        )
    axis.set(
        xlabel="Frequency (GHz)",
        ylabel="S11 magnitude (dB)",
        title="SAB-derived patch antenna input reflection",
    )
    axis.set_xlim(1.6, 3.2)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(HERE / "patch_s11.png", dpi=180)
    plt.close(figure)
    return minima


def main() -> None:
    series = {
        "gprMax": read_gprmax_s11(),
        "CST FIT": read_touchstone_s11(CST_FIT_S11),
        "CST FEM": read_touchstone_s11(CST_FEM_S11),
    }
    minima = plot_s11(series)
    gprmax_frequency, gprmax_s11 = minima["gprMax"]
    cst_fit_frequency, cst_fit_s11 = minima["CST FIT"]
    cst_fem_frequency, cst_fem_s11 = minima["CST FEM"]
    summary = {
        "gprmax_s11_minimum_frequency_hz": gprmax_frequency,
        "gprmax_s11_minimum_db": gprmax_s11,
        "cst_fit_s11_minimum_frequency_hz": cst_fit_frequency,
        "cst_fit_s11_minimum_db": cst_fit_s11,
        "cst_fem_s11_minimum_frequency_hz": cst_fem_frequency,
        "cst_fem_s11_minimum_db": cst_fem_s11,
        "gprmax_minus_cst_fit_minimum_frequency_hz": gprmax_frequency
        - cst_fit_frequency,
        "gprmax_minus_cst_fit_minimum_frequency_percent": 100.0
        * (gprmax_frequency - cst_fit_frequency)
        / cst_fit_frequency,
        "gprmax_minus_cst_fit_minimum_depth_db": gprmax_s11 - cst_fit_s11,
        "gprmax_minus_cst_fem_minimum_frequency_hz": gprmax_frequency
        - cst_fem_frequency,
        "gprmax_minus_cst_fem_minimum_frequency_percent": 100.0
        * (gprmax_frequency - cst_fem_frequency)
        / cst_fem_frequency,
        "gprmax_minus_cst_fem_minimum_depth_db": gprmax_s11 - cst_fem_s11,
        "cst_fit_minus_cst_fem_minimum_frequency_hz": cst_fit_frequency
        - cst_fem_frequency,
        "cst_fit_minus_cst_fem_minimum_depth_db": cst_fit_s11 - cst_fem_s11,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
