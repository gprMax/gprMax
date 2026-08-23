"""Plot multi-material results from ``validate_sar_2d_cylinder`` outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODES = ("TM", "TE")


def _parse_case(value):
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cases must have the form NAME=PATH") from exc
    return name, Path(directory)


def _find_result(directory, material, mode):
    patterns = (
        f"sar_2d_cylinder_{material}_{mode.lower()}_*.npz",
        f"sar_2d_cylinder_{mode.lower()}_*.npz",
    )
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple result files match {directory / pattern}")
    raise FileNotFoundError(f"no {mode} result found in {directory}")


def _find_summary(directory, material):
    patterns = (
        f"sar_2d_cylinder_{material}_*_summary.json",
        "sar_2d_cylinder_*_summary.json",
    )
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple summary files match {directory / pattern}")
    raise FileNotFoundError(f"no summary found in {directory}")


def load_cases(case_arguments):
    cases = {}
    for material, directory in case_arguments:
        summary = json.loads(_find_summary(directory, material).read_text())
        modes = {}
        for mode in MODES:
            with np.load(_find_result(directory, material, mode)) as result:
                modes[mode] = {
                    "cells": np.asarray(result["cells"]),
                    "numerical": np.asarray(result["numerical_sar"]),
                    "analytical": np.asarray(result["analytical_collocated_sar"]),
                    "metrics": summary[mode],
                }
        cases[material] = modes
    return cases


def _raster(data, values):
    cells = data["cells"]
    i0, j0 = np.min(cells[:, :2], axis=0)
    i1, j1 = np.max(cells[:, :2], axis=0)
    image = np.full((j1 - j0 + 1, i1 - i0 + 1), np.nan)
    image[cells[:, 1] - j0, cells[:, 0] - i0] = values
    dl = data["metrics"]["dl_m"]
    x = (cells[:, 0] + 0.5) * dl
    y = (cells[:, 1] + 0.5) * dl
    centre_x = 0.5 * (np.min(x) + np.max(x))
    centre_y = 0.5 * (np.min(y) + np.max(y))
    extent = (
        (np.min(x) - centre_x - 0.5 * dl) * 1e3,
        (np.max(x) - centre_x + 0.5 * dl) * 1e3,
        (np.min(y) - centre_y - 0.5 * dl) * 1e3,
        (np.max(y) - centre_y + 0.5 * dl) * 1e3,
    )
    return image, extent


def plot_maps(cases, output_dir):
    rows = len(cases) * len(MODES)
    figure, axes = plt.subplots(rows, 3, figsize=(10.8, 3.35 * rows))
    map_image = None
    error_image = None
    row = 0
    for material, modes in cases.items():
        for mode in MODES:
            data = modes[mode]
            peak = np.max(data["analytical"])
            exact_db = 10 * np.log10(np.maximum(data["analytical"] / peak, 1e-3))
            numerical_db = 10 * np.log10(np.maximum(data["numerical"] / peak, 1e-3))
            error = 100 * np.abs(data["numerical"] - data["analytical"]) / peak
            for column, (values, title) in enumerate(
                (
                    (exact_db, "Exact SAR / exact peak [dB]"),
                    (numerical_db, "gprMax SAR / exact peak [dB]"),
                    (error, "Absolute error / exact peak [%]"),
                )
            ):
                image, extent = _raster(data, values)
                if column < 2:
                    plotted = axes[row, column].imshow(
                        image,
                        origin="lower",
                        extent=extent,
                        cmap="inferno",
                        vmin=-30,
                        vmax=0,
                    )
                    map_image = plotted
                else:
                    plotted = axes[row, column].imshow(
                        image,
                        origin="lower",
                        extent=extent,
                        cmap="magma",
                        vmin=0,
                        vmax=10,
                    )
                    error_image = plotted
                axes[row, column].set_aspect("equal")
                axes[row, column].set_title(title)
                axes[row, column].set_xlabel("x relative to centre [mm]")
                axes[row, column].set_ylabel("y relative to centre [mm]")
            axes[row, 0].text(
                0.02,
                0.98,
                f"{material}, {mode}z",
                transform=axes[row, 0].transAxes,
                va="top",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
            )
            row += 1
    figure.subplots_adjust(right=0.9, hspace=0.35, wspace=0.28)
    map_bar = figure.add_axes((0.92, 0.53, 0.015, 0.35))
    error_bar = figure.add_axes((0.92, 0.12, 0.015, 0.35))
    figure.colorbar(map_image, cax=map_bar, label="Normalized SAR [dB]")
    figure.colorbar(error_image, cax=error_bar, label="Error / exact peak [%]")
    figure.savefig(output_dir / "sar_2d_cylinder_material_maps.png", dpi=220)
    plt.close(figure)


def plot_boundary_profiles(cases, output_dir, maximum_depth_cells=30):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    markers = ("o", "s", "^")
    for mode, axis in zip(MODES, axes):
        for marker, (material, modes) in zip(markers, cases.items()):
            data = modes[mode]
            cells = data["cells"]
            dl = data["metrics"]["dl_m"]
            radius = data["metrics"]["radius_m"]
            coordinates = (cells[:, :2] + 0.5) * dl
            centre = 0.5 * (np.min(coordinates, axis=0) + np.max(coordinates, axis=0))
            depth = (radius - np.linalg.norm(coordinates - centre, axis=1)) / dl
            bins = np.floor(np.maximum(depth, 0)).astype(int)
            profile = []
            centres = []
            for index in range(maximum_depth_cells):
                selected = bins == index
                if not np.any(selected):
                    continue
                difference = data["numerical"][selected] - data["analytical"][selected]
                denominator = np.linalg.norm(data["analytical"][selected])
                if denominator == 0:
                    continue
                centres.append(index + 0.5)
                profile.append(100 * np.linalg.norm(difference) / denominator)
            axis.plot(
                centres,
                profile,
                color="black",
                marker=marker,
                markerfacecolor="none",
                markevery=3,
                label=material,
            )
        axis.set(
            xlabel="Depth inward from exact boundary [cells]",
            title=f"{mode}z",
            xlim=(0, maximum_depth_cells),
        )
        axis.grid(True, color="0.85", linewidth=0.6)
        axis.legend()
    axes[0].set_ylabel("Shell relative L2 SAR error [%]")
    figure.tight_layout()
    figure.savefig(output_dir / "sar_2d_cylinder_boundary_error.png", dpi=220)
    plt.close(figure)


def plot_summary(cases, output_dir):
    labels = []
    local_errors = []
    power_errors = []
    for material, modes in cases.items():
        for mode in MODES:
            metrics = modes[mode]["metrics"]
            labels.append(f"{material}\n{mode}z")
            local_errors.append(100 * metrics["local_sar"]["interior_above_5_percent_peak"]["relative_l2_error"])
            power_errors.append(100 * metrics["relative_absorbed_power_error"])

    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for axis, values, title in (
        (axes[0], local_errors, "Interior local SAR relative L2 error"),
        (axes[1], power_errors, "Absorbed power per unit length error"),
    ):
        axis.bar(x, values, facecolor="white", edgecolor="black", hatch="//")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Relative error [%]")
        axis.set_title(title)
        axis.grid(True, axis="y", color="0.85", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_dir / "sar_2d_cylinder_error_summary.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        help="material name and validation directory as NAME=PATH; repeat as needed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.case)
    plot_maps(cases, args.output_dir)
    plot_boundary_profiles(cases, args.output_dir)
    plot_summary(cases, args.output_dir)


if __name__ == "__main__":
    main()
