# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Export a transparent logo master at standard print and screen widths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

DEFAULT_WIDTHS = (2048, 1024, 512, 400, 256)
BACKGROUNDS = {"white": "#ffffff", "dark": "#111827"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(master: Path, output_dir: Path, widths: tuple[int, ...]) -> dict[str, object]:
    """Export all requested sizes and return their machine-readable manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(master).convert("RGBA")
    unique_widths = sorted(set(widths), reverse=True)
    if not unique_widths or min(unique_widths) < 1:
        raise ValueError("At least one positive output width is required")
    if unique_widths[0] > source.width:
        raise ValueError(
            f"Largest requested width {unique_widths[0]} exceeds master width {source.width}"
        )

    manifest: dict[str, object] = {
        "source": master.name,
        "source_pixels": list(source.size),
        "print_dpi": 300,
        "assets": [],
    }
    assets = manifest["assets"]
    assert isinstance(assets, list)
    for width in unique_widths:
        height = round(source.height * width / source.width)
        image = (
            source.copy()
            if width == source.width
            else source.resize((width, height), Image.Resampling.LANCZOS)
        )
        transparent = output_dir / f"gprmax_v4_logo_{width}px.png"
        image.save(transparent, dpi=(300, 300), optimize=True)
        assets.append(
            {
                "file": transparent.name,
                "background": "transparent",
                "pixels": [width, height],
                "print_width_cm_at_300dpi": width / 300 * 2.54,
                "sha256": digest(transparent),
            }
        )
        for suffix, colour in BACKGROUNDS.items():
            background = Image.new("RGBA", image.size, colour)
            background.alpha_composite(image)
            output = output_dir / f"gprmax_v4_logo_{width}px_on_{suffix}.png"
            background.convert("RGB").save(output, dpi=(300, 300), optimize=True)
            assets.append(
                {
                    "file": output.name,
                    "background": suffix,
                    "pixels": [width, height],
                    "print_width_cm_at_300dpi": width / 300 * 2.54,
                    "sha256": digest(output),
                }
            )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    args = parser.parse_args()
    print(json.dumps(export(args.master, args.output_dir, tuple(args.widths)), indent=2))


if __name__ == "__main__":
    main()
