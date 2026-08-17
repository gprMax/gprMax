# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Render the resonant gprMax logo from a 2D TMz field snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

SITE_PURPLE = "#55037f"
BRAND_CMAP = LinearSegmentedColormap.from_list(
    "gprmax_brand",
    (
        (0.00, "#28367f"),
        (0.22, "#1656b8"),
        (0.43, "#3b8cff"),
        (0.60, SITE_PURPLE),
        (0.80, "#c42b91"),
        (1.00, "#ff9bca"),
    ),
)


def free_space_mask(
    input_file: Path, shape: tuple[int, int], spacing: tuple[float, float]
) -> np.ndarray:
    """Rebuild the carved-letter mask from the compact box commands."""
    mask = np.zeros(shape, dtype=bool)
    dx, dy = spacing
    for line in input_file.read_text().splitlines():
        if not line.startswith("#box:") or "free_space" not in line:
            continue
        tokens = line.split()
        x0, y0 = round(float(tokens[1]) / dx), round(float(tokens[2]) / dy)
        x1, y1 = round(float(tokens[4]) / dx), round(float(tokens[5]) / dy)
        mask[x0:x1, y0:y1] = True
    return mask


def read_snapshot(snapshot: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Read the invariant Ez plane and its in-plane spatial steps."""
    with h5py.File(snapshot) as output:
        field = np.asarray(output["Ez"])
        spacing = np.asarray(output.attrs["dx_dy_dz"], dtype=float)
    field = np.squeeze(field)
    if field.ndim != 2:
        raise ValueError(f"Expected one invariant Ez plane, received shape {field.shape}")
    return field, (float(spacing[0]), float(spacing[1]))


def render(input_file: Path, snapshot: Path, output: Path, width_px: int = 2048) -> None:
    """Render the official transparent, field-filled wordmark."""
    if width_px < 1:
        raise ValueError("width_px must be positive")
    field, spacing = read_snapshot(snapshot)
    mask = free_space_mask(input_file, field.shape, spacing)
    if not np.any(mask):
        raise ValueError("No carved free-space cells were found in the input model")

    magnitude = np.abs(field)
    scale = float(np.percentile(magnitude[mask], 99.5))
    normalised = np.clip(magnitude / scale, 0, 1)
    nx, ny = field.shape
    extent = (0, nx * spacing[0], 0, ny * spacing[1])
    occupied_x, occupied_y = np.where(mask)
    margin = 0.01
    crop = (
        max(0, occupied_x.min() * spacing[0] - margin),
        min(extent[1], (occupied_x.max() + 1) * spacing[0] + margin),
        max(0, occupied_y.min() * spacing[1] - margin),
        min(extent[3], (occupied_y.max() + 1) * spacing[1] + margin),
    )
    aspect = (crop[3] - crop[2]) / (crop[1] - crop[0])
    height_px = round(width_px * aspect)
    dpi = 100
    figure, axis = plt.subplots(figsize=(width_px / dpi, height_px / dpi))
    figure.patch.set_alpha(0)
    axis.patch.set_alpha(0)

    plotted = np.ma.array(normalised.T, mask=~mask.T)
    axis.imshow(
        plotted,
        origin="lower",
        extent=extent,
        cmap=BRAND_CMAP,
        norm=PowerNorm(0.42, vmin=0, vmax=1),
        interpolation="bilinear",
    )
    axis.contour(
        mask.T.astype(float),
        levels=(0.5,),
        colors="#f7f0fa",
        linewidths=1.15,
        origin="lower",
        extent=extent,
    )
    axis.contour(
        mask.T.astype(float),
        levels=(0.5,),
        colors=SITE_PURPLE,
        linewidths=0.35,
        origin="lower",
        extent=extent,
    )
    axis.set_xlim(crop[0], crop[1])
    # PIL and the snapshot use opposite display conventions for y.
    axis.set_ylim(crop[3], crop[2])
    axis.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, transparent=True, pad_inches=0)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width-px", type=int, default=2048)
    args = parser.parse_args()
    render(args.input_file, args.snapshot, args.output, args.width_px)


if __name__ == "__main__":
    main()
