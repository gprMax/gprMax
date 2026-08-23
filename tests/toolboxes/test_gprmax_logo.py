"""Tests for the reproducible gprMax version 4 logo toolbox."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from toolboxes.gprMaxLogo.export_assets import export
from toolboxes.gprMaxLogo.logo_model import SOURCE_SPECS, rectangles
from gprMax.hash_cmds_file import check_cmd_names

ROOT = Path(__file__).resolve().parents[2]
LOGO_TOOLBOX = ROOT / "toolboxes" / "gprMaxLogo"


def test_official_model_and_metadata_are_consistent():
    metadata = json.loads((LOGO_TOOLBOX / "model" / "gprmax_v4_logo.json").read_text())
    manifest = json.loads((LOGO_TOOLBOX / "assets" / "manifest.json").read_text())
    lines = (LOGO_TOOLBOX / "model" / "gprmax_v4_logo.in").read_text().splitlines()
    free_space_boxes = [line for line in lines if line.startswith("#box:") and "free_space" in line]
    sources = [line for line in lines if line.startswith("#hertzian_dipole:")]

    assert metadata["grid_cells"] == [12000, 6000]
    assert metadata["free_space_cells"] == 10_666_982
    assert metadata["rectangles"] == len(free_space_boxes) == 9724
    assert len(metadata["sources"]) == len(sources) == 8
    assert metadata["source_model"]["waveform_quantity"] == "line current (A)"
    assert metadata["source_model"]["grid_policy"] == "single authoritative model"
    assert metadata["brand_master"] is True
    assert [source["amplitude"] for source in metadata["sources"]] == [
        spec[3] for spec in SOURCE_SPECS
    ]
    assert all(len(source.split()) == 8 for source in sources)
    assert lines[-1].endswith("10e-9 logo_fields.h5")

    commands = [f"{line}\n" for line in lines if line.startswith("#")]
    single, multi, geometry = check_cmd_names(commands)
    assert single["#domain_mode"] == "TM"
    assert len(multi["#hertzian_dipole"]) == 8
    assert len(geometry) == 9725  # One PEC plane and 9724 free-space boxes.
    assert manifest["source"] == "gprmax_v4_logo_master_10000.png"
    assert manifest["source_pixels"] == [10000, 2788]
    assert {asset["pixels"][0] for asset in manifest["assets"]} == {
        2048,
        1024,
        512,
        400,
        256,
    }
    assert (ROOT / "images_shared" / "gprMax_logo.png").read_bytes() == (
        LOGO_TOOLBOX / "assets" / "gprmax_v4_logo_1024px.png"
    ).read_bytes()
    assert (ROOT / "images_shared" / "gprMax_logo_small.png").read_bytes() == (
        LOGO_TOOLBOX / "assets" / "gprmax_v4_logo_400px.png"
    ).read_bytes()


def test_rectangle_compression_reconstructs_mask():
    mask = np.zeros((9, 7), dtype=bool)
    mask[1:5, 1:3] = True
    mask[2:8, 3:6] = True

    rebuilt = np.zeros_like(mask)
    for x0, y0, x1, y1 in rectangles(mask):
        rebuilt[x0:x1, y0:y1] = True

    np.testing.assert_array_equal(rebuilt, mask)


def test_standard_asset_export(tmp_path):
    master = tmp_path / "master.png"
    Image.new("RGBA", (100, 30), (85, 3, 127, 255)).save(master)

    manifest = export(master, tmp_path / "assets", (50, 25))

    assert len(manifest["assets"]) == 6
    assert Image.open(tmp_path / "assets" / "gprmax_v4_logo_50px.png").size == (50, 15)
    assert Image.open(tmp_path / "assets" / "gprmax_v4_logo_25px_on_dark.png").mode == "RGB"
