"""Measure internal-PML termination reflection versus slab thickness.

The experiment launches the fundamental mode of a straight PEC rectangular
waveguide toward an internal x-minus PML.  The PML entrance, source plane, and
waveguide cross-section stay fixed while the PEC-backed slab grows toward x=0.
This avoids mixing the length trend with a changing source-to-load distance.

Two fills are exercised:

* ``empty`` is the conservative low-loss case, where the backing reflection is
  not attenuated by the guide material.
* ``lossy_debye`` proves that the internal PML can operate with the same
  conductivity and Debye ADE used by the broadband FDFD eigenmode source.

The longest requested slab is used as a numerical reference.  In addition to
raw S11, the report gives ``S11 - S11_reference`` to remove the common finite-
time/source residual and reveal excess reflection from a shorter slab.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DX = 0.0002
PML_ENTRANCE_X = 0.012
SOURCE_X = 0.018
FREQUENCY_MIN = 30e9
FREQUENCY_MAX = 45e9

MATERIAL_LINES = {
    "empty": (),
    "lossy_debye": (
        "#material: 2.5 0.01 1 0 guide_fill",
        "#add_dispersion_debye: 1 2.5 2.5e-12 guide_fill",
        "#box: 0 0.001 0.001 0.030 0.007 0.005 guide_fill",
    ),
}


def _model_text(material: str, cells: int) -> str:
    lines = [
        f"#title: Internal PML length sweep ({material}, {cells} cells)",
        "",
        "#domain: 0.030 0.008 0.006",
        f"#dx_dy_dz: {DX:g} {DX:g} {DX:g}",
        "#time_window: 1.2e-9",
        "#pml_cells: 0 0 0 10 0 0",
        "",
        *MATERIAL_LINES[material],
        "",
        "#box: 0 0 0 0.030 0.001 0.006 pec",
        "#box: 0 0.007 0 0.030 0.008 0.006 pec",
        "#box: 0 0 0 0.030 0.008 0.001 pec",
        "#box: 0 0 0.005 0.030 0.008 0.006 pec",
        "",
    ]
    if cells:
        backing_x = PML_ENTRANCE_X - cells * DX
        if backing_x < 0:
            raise ValueError(f"{cells} cells places the backing outside the domain")
        lines.append(
            "#pml_slab: "
            f"{backing_x:.10g} 0.001 0.001 {PML_ENTRANCE_X:g} 0.007 0.005 x0"
        )
    else:
        # A fully reflecting control at the same plane as every PML entrance.
        lines.append(
            f"#plate: {PML_ENTRANCE_X:g} 0.001 0.001 "
            f"{PML_ENTRANCE_X:g} 0.007 0.005 pec"
        )
    lines.extend(
        [
            "",
            f"#eigenmode_band: waveguide_band {FREQUENCY_MIN:g} {FREQUENCY_MAX:g} 61",
            f"#eigenmode_port: 1 {SOURCE_X:g} 0.001 0.001 {SOURCE_X:g} 0.007 0.005 - 1 auto y",
            "#eigenmode_excitation: 1 1 auto y",
            "",
        ]
    )
    return "\n".join(lines)


def _case_name(material: str, cells: int) -> str:
    suffix = "pec_control" if cells == 0 else f"pml_{cells:02d}_cells"
    return f"{material}_{suffix}"


def _run_case(output_dir: Path, material: str, cells: int, force: bool) -> Path:
    name = _case_name(material, cells)
    input_path = output_dir / "inputs" / f"{name}.in"
    stem = output_dir / "runs" / name
    csv_path = stem.with_name(stem.name + "_sparameters.csv")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    stem.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(_model_text(material, cells), encoding="utf-8")
    if csv_path.exists() and not force:
        print(f"Reusing {csv_path}")
        return csv_path

    log_path = stem.with_suffix(".log")
    print(f"Running {material}, {cells} cells")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gprMax",
                str(input_path),
                "-outputfile",
                str(stem),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not csv_path.exists():
        raise FileNotFoundError(f"gprMax did not write {csv_path}; see {log_path}")
    return csv_path


def _read_s11(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["destination_port"]) == 1
            and int(row["destination_mode"]) == 1
            and int(row["power_wave_valid"])
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if not rows:
        raise ValueError(f"No valid mode-1 S11 rows in {path}")
    frequency = np.asarray([float(row["frequency_hz"]) for row in rows])
    s11 = np.asarray(
        [complex(float(row["S_real"]), float(row["S_imag"])) for row in rows]
    )
    return frequency, s11


def _db(values: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def _analyse(
    output_dir: Path,
    materials: list[str],
    lengths: list[int],
    paths: dict[tuple[str, int], Path],
) -> tuple[Path, Path, Path, Path]:
    reference_cells = max(lengths)
    detailed_path = output_dir / "pml_length_sweep.csv"
    summary_path = output_dir / "pml_length_summary.json"
    report_path = output_dir / "pml_length_report.md"
    plot_path = output_dir / "pml_length_sweep.png"
    summary: dict[str, list[dict[str, float | int]]] = {}

    with detailed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "material",
                "length_cells",
                "length_mm",
                "frequency_hz",
                "s11_real",
                "s11_imag",
                "s11_db",
                "excess_vs_long_reference_db",
            )
        )
        for material in materials:
            reference_frequency, reference_s11 = _read_s11(
                paths[(material, reference_cells)]
            )
            material_summary = []
            for cells in lengths:
                frequency, s11 = _read_s11(paths[(material, cells)])
                np.testing.assert_allclose(frequency, reference_frequency, rtol=0, atol=1)
                excess = s11 - reference_s11
                raw_db = _db(s11)
                excess_db = _db(excess)
                for values in zip(frequency, s11.real, s11.imag, raw_db, excess_db):
                    writer.writerow((material, cells, cells * DX * 1e3, *values))
                material_summary.append(
                    {
                        "length_cells": cells,
                        "length_mm": cells * DX * 1e3,
                        "worst_s11_db": float(np.max(raw_db)),
                        "median_s11_db": float(np.median(raw_db)),
                        "worst_excess_vs_long_db": float(np.max(excess_db)),
                        "median_excess_vs_long_db": float(np.median(excess_db)),
                    }
                )
            summary[material] = material_summary

    summary_path.write_text(
        json.dumps(
            {
                "reference_length_cells": reference_cells,
                "cell_size_mm": DX * 1e3,
                "frequency_band_ghz": [FREQUENCY_MIN * 1e-9, FREQUENCY_MAX * 1e-9],
                "results": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "# Internal PML waveguide length sweep",
        "",
        f"Band: {FREQUENCY_MIN * 1e-9:g}-{FREQUENCY_MAX * 1e-9:g} GHz; "
        f"cell size: {DX * 1e3:g} mm; longest-slab reference: {reference_cells} cells.",
        "",
        "`Worst excess` is the maximum over the band of "
        "|S11(length) - S11(reference)|. The reference row is omitted because "
        "its self-difference is exactly zero.",
        "",
    ]
    for material in materials:
        report.extend(
            [
                f"## {material}",
                "",
                "| Cells | Length (mm) | Worst raw S11 (dB) | Worst excess (dB) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in summary[material]:
            cells = int(row["length_cells"])
            if cells == reference_cells:
                excess = "reference"
            else:
                excess = f"{float(row['worst_excess_vs_long_db']):.2f}"
            report.append(
                f"| {cells} | {float(row['length_mm']):.1f} | "
                f"{float(row['worst_s11_db']):.2f} | {excess} |"
            )
        report.append("")
    report_path.write_text("\n".join(report), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for material in materials:
        records = summary[material]
        length_mm = [float(row["length_mm"]) for row in records]
        axes[0].plot(
            length_mm,
            [float(row["worst_s11_db"]) for row in records],
            marker="o",
            label=material,
        )
        axes[1].plot(
            length_mm[:-1],
            [float(row["worst_excess_vs_long_db"]) for row in records[:-1]],
            marker="o",
            label=material,
        )
    axes[0].set_title("Worst raw S11 over 30-45 GHz")
    axes[1].set_title(f"Worst excess reflection vs {reference_cells}-cell slab")
    for axis in axes:
        axis.set_xlabel("PML length (mm)")
        axis.set_ylabel("Magnitude (dB)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[1].set_ylim(-150, 5)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return detailed_path, summary_path, report_path, plot_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "length_sweep_results",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        choices=tuple(MATERIAL_LINES),
        default=list(MATERIAL_LINES),
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[0, 4, 6, 8, 10, 12, 16, 20, 30, 40, 50],
        help="PML lengths in cells; zero is a PEC control.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    lengths = sorted(set(args.lengths))
    if not lengths or lengths[-1] == 0:
        parser.error("at least one non-zero PML length is required")
    if any(cells < 0 for cells in lengths):
        parser.error("PML lengths cannot be negative")

    output_dir = args.output_dir.resolve()
    paths = {
        (material, cells): _run_case(output_dir, material, cells, args.force)
        for material in args.materials
        for cells in lengths
    }
    outputs = _analyse(output_dir, args.materials, lengths, paths)
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
