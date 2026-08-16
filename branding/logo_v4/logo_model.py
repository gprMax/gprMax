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
BASE_DL = 0.00025
BASE_NX = 4000
BASE_NY = 2000
BASE_FONT_SIZE = 996
OFFICIAL_REFINEMENT = 3
FONT = Path(__file__).resolve().parent / "fonts" / "IBMPlexSans-Bold.ttf"
# Subpixel glyph advances are rounded by Pillow at draw time. These one-pixel
# offsets preserve the exact raster used for the approved 3x master rather
# than allowing a future Pillow rounding change to move most glyphs by one
# 0.083 mm cell.
OFFICIAL_GLYPH_X_OFFSETS = (1, 1, 1, 1, 1, 0)

# Source locations were selected in the deeply interior parts of the glyphs on
# the base grid. Scaling the indices preserves their physical locations on the
# official 3x grid. Amplitudes were calibrated on that grid so that the RMS
# field in each isolated glyph is approximately equal.
SOURCE_SPECS = (
    ("g1", "g", (712, 1224), 2.076331, 10e9, 5e-9),
    ("g2", "g", (367, 877), 2.076331, 10e9, 5e-9),
    ("p1", "p", (948, 845), 2.974881, 10e9, 5e-9),
    ("r1", "r", (1555, 872), 1.268562, 10e9, 5e-9),
    ("M1", "M", (2439, 682), 0.941482, 12e9, 7e-9),
    ("M2", "M", (1993, 677), 0.941482, 12e9, 7e-9),
    ("a1", "a", (3033, 979), 2.053664, 12e9, 3e-9),
    ("x1", "x", (3470, 971), 0.898925, 12e9, 5e-9),
)


def text_mask(refinement: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Rasterise the IBM Plex Sans wordmark directly on an FDTD grid."""
    if refinement < 1:
        raise ValueError("refinement must be a positive integer")
    if not FONT.exists():
        raise FileNotFoundError(f"The bundled IBM Plex Sans font was not found: {FONT}")

    nx, ny = BASE_NX * refinement, BASE_NY * refinement
    font = ImageFont.truetype(FONT, BASE_FONT_SIZE * refinement)
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
    offsets = OFFICIAL_GLYPH_X_OFFSETS if refinement == OFFICIAL_REFINEMENT else (0,) * len(TEXT)
    for character, advance, offset in zip(TEXT, advances, offsets):
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


def sources(refinement: int, mask: np.ndarray) -> list[dict[str, object]]:
    """Return the calibrated source definitions for a refined grid."""
    result: list[dict[str, object]] = []
    for name, glyph, base_point, amplitude, frequency, stop in SOURCE_SPECS:
        point = (base_point[0] * refinement, base_point[1] * refinement)
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
    refinement: int,
    mask: np.ndarray,
    source_definitions: list[dict[str, object]],
) -> dict[str, object]:
    """Write the complete hash-command model and return its metadata."""
    dl = BASE_DL / refinement
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
        "grid_cells": [BASE_NX * refinement, BASE_NY * refinement],
        "physical_domain_m": [1.0, 0.5],
        "free_space_cells": int(mask.sum()),
        "rectangles": len(boxes),
        "sources": source_definitions,
    }


def generate(refinement: int, output_dir: Path) -> dict[str, object]:
    """Generate a model, its geometry preview, and portable metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask, glyphs = text_mask(refinement)
    source_definitions = sources(refinement, mask)
    stem = f"gprmax_v4_logo_{refinement}x"
    metadata = write_model(output_dir / f"{stem}.in", refinement, mask, source_definitions)
    Image.fromarray((mask.T * 255).astype(np.uint8)).save(output_dir / f"{stem}_geometry.png")
    metadata.update(
        {
            "refinement": refinement,
            "font_file": "fonts/IBMPlexSans-Bold.ttf",
            "font_size_pixels": BASE_FONT_SIZE * refinement,
            "glyphs": glyphs,
            "amplitude_calibration": (
                "Per-glyph RMS field calibrated on the official 3x model"
                if refinement == OFFICIAL_REFINEMENT
                else "Official 3x amplitudes retained; this is not an approved brand master"
            ),
        }
    )
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=int, default=OFFICIAL_REFINEMENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.refinement, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
