# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate the FDTD model used for the gprMax version 4 logo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXT = "gprMax"
GRID_SPACING = 0.00025 / 3
GRID_CELLS = (12000, 6000)
FONT_SIZE = 2988
FONT = Path(__file__).resolve().parent / "fonts" / "IBMPlexSans-Bold.ttf"
# Subpixel glyph advances are rounded by Pillow at draw time. Keep an explicit
# per-glyph correction table so the approved raster can be pinned if a future
# Pillow release changes its rounding. No corrections are required with the
# bundled font and currently tested Pillow version.
GLYPH_X_OFFSETS = (0, 0, 0, 0, 0, 0)

# Source locations were selected in deeply interior parts of the glyphs.
# Amplitudes were calibrated on this fixed grid so that the RMS field in each
# isolated glyph is approximately equal.
SOURCE_SPECS = (
    ("g1", "g", (2136, 3672), 2.076331, 10e9, 5e-9),
    ("g2", "g", (1101, 2631), 2.076331, 10e9, 5e-9),
    ("p1", "p", (2844, 2535), 2.974881, 10e9, 5e-9),
    ("r1", "r", (4665, 2616), 1.268562, 10e9, 5e-9),
    ("M1", "M", (7317, 2046), 0.941482, 12e9, 7e-9),
    ("M2", "M", (5979, 2031), 0.941482, 12e9, 7e-9),
    ("a1", "a", (9099, 2937), 2.053664, 12e9, 3e-9),
    ("x1", "x", (10410, 2913), 0.898925, 12e9, 5e-9),
)


def text_mask() -> tuple[np.ndarray, list[dict[str, object]]]:
    """Rasterise the IBM Plex Sans wordmark on the authoritative FDTD grid."""
    if not FONT.exists():
        raise FileNotFoundError(f"The bundled IBM Plex Sans font was not found: {FONT}")

    nx, ny = GRID_CELLS
    font = ImageFont.truetype(FONT, FONT_SIZE)
    advances = [float(font.getlength(character)) for character in TEXT]
    boxes = [font.getbbox(character, anchor="ls") for character in TEXT]
    width = sum(advances)
    top = min(box[1] for box in boxes)
    bottom = max(box[3] for box in boxes)

    image = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(image)
    x = (nx - width) / 2
    baseline = (ny - (bottom - top)) / 2 - top
    glyphs: list[dict[str, object]] = []
    for character, advance, offset in zip(TEXT, advances, GLYPH_X_OFFSETS):
        before = np.asarray(image).copy()
        draw.text((x + offset, baseline), character, font=font, fill=255, anchor="ls")
        added = np.asarray(image) > before
        rows, columns = np.where(added)
        glyphs.append(
            {
                "character": character,
                "bounds_indices": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max() + 1),
                    int(rows.max() + 1),
                ],
            }
        )
        x += advance

    # PIL row indices increase downwards. They are deliberately mapped to the
    # FDTD y index; render_logo.py performs the display reflection.
    return (np.asarray(image) >= 128).T, glyphs


def rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Compress a binary cell mask into exact, vertically extended runs."""
    active: dict[tuple[int, int], int] = {}
    result: list[tuple[int, int, int, int]] = []
    for y in range(mask.shape[1] + 1):
        xs = np.flatnonzero(mask[:, y]) if y < mask.shape[1] else np.empty(0, dtype=int)
        runs: list[tuple[int, int]] = []
        if xs.size:
            split = np.where(np.diff(xs) != 1)[0] + 1
            runs = [(int(group[0]), int(group[-1] + 1)) for group in np.split(xs, split)]
        current = set(runs)
        for run, y0 in list(active.items()):
            if run not in current:
                result.append((run[0], y0, run[1], y))
                del active[run]
        for run in sorted(current):
            active.setdefault(run, y)
    return result


def number(value: float) -> str:
    """Format an input-file number without unnecessary decimal noise."""
    return f"{value:.12g}"


def sources(mask: np.ndarray) -> list[dict[str, object]]:
    """Return the calibrated source definitions for the fixed logo model."""
    result: list[dict[str, object]] = []
    for name, glyph, point, amplitude, frequency, stop in SOURCE_SPECS:
        if not mask[point]:
            raise ValueError(f"Source {name} at {point} lies outside its glyph")
        result.append(
            {
                "name": name,
                "glyph": glyph,
                "point": list(point),
                "amplitude": amplitude,
                "frequency": frequency,
                "start": 0,
                "stop": stop,
            }
        )
    return result


def write_model(
    output: Path,
    mask: np.ndarray,
    source_definitions: list[dict[str, object]],
) -> dict[str, object]:
    """Write the complete hash-command model and return its metadata."""
    dl = GRID_SPACING
    boxes = rectangles(mask)
    lines = [
        "#title: gprMax v4 IBM Plex Sans resonant-field logo",
        "#domain_mode: TM",
        "#domain: 1 0.5 inf",
        f"#dx_dy_dz: {number(dl)} {number(dl)} {number(dl)}",
        "#time_window: 10.05e-9",
        "#pml_cells: 0",
        "",
    ]
    for source in source_definitions:
        lines.append(
            f"#waveform: contsine {number(float(source['amplitude']))} "
            f"{number(float(source['frequency']))} logo_{source['name']}"
        )
    lines.append("")
    for source in source_definitions:
        x, y = source["point"]
        lines.append(
            f"#hertzian_dipole: z {number(int(x) * dl)} {number(int(y) * dl)} inf "
            f"logo_{source['name']} {number(float(source['start']))} "
            f"{number(float(source['stop']))}"
        )

    lines.extend(("", "#box: 0 0 0 1 0.5 inf pec"))
    for x0, y0, x1, y1 in boxes:
        lines.append(
            f"#box: {number(x0 * dl)} {number(y0 * dl)} 0 "
            f"{number(x1 * dl)} {number(y1 * dl)} inf free_space n"
        )
    lines.extend(
        (
            "",
            f"#snapshot: 0 0 0 1 0.5 inf {number(dl)} {number(dl)} {number(dl)} "
            "10e-9 logo_fields.h5",
            "",
        )
    )
    output.write_text("\n".join(lines))

    return {
        "input_file": output.name,
        "grid_spacing_m": dl,
        "grid_cells": list(GRID_CELLS),
        "physical_domain_m": [1.0, 0.5],
        "free_space_cells": int(mask.sum()),
        "rectangles": len(boxes),
        "sources": source_definitions,
    }


def generate(output_dir: Path) -> dict[str, object]:
    """Generate the authoritative model, geometry preview, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask, glyphs = text_mask()
    source_definitions = sources(mask)
    stem = "gprmax_v4_logo"
    metadata = write_model(output_dir / f"{stem}.in", mask, source_definitions)
    Image.fromarray((mask.T * 255).astype(np.uint8)).save(output_dir / f"{stem}_geometry.png")
    metadata.update(
        {
            "font_file": "fonts/IBMPlexSans-Bold.ttf",
            "font_size_pixels": FONT_SIZE,
            "glyphs": glyphs,
            "source_model": {
                "waveform_quantity": "line current (A)",
                "grid_policy": "single authoritative model",
            },
            "amplitude_calibration": "Per-glyph visual balance calibrated on this model",
            "brand_master": True,
        }
    )
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
