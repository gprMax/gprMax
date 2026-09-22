"""Physical reflection and transmission regressions for tracking examples."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.sources import EigenmodeSource


@pytest.mark.integration
@pytest.mark.parametrize("mode", (1, 2))
@pytest.mark.parametrize(
    "example",
    (
        "example_8_auto_degenerate_te11/auto_degenerate_te11",
        "example_9_auto_mode_crossing/auto_mode_crossing",
    ),
)
def test_tracking_example_straight_guide(tmp_path, monkeypatch, example, mode, record_property):
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/features/eigenmode_ports" / f"{example}.py"
    spec = importlib.util.spec_from_file_location("tracking_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = FDTDGrid.build

    def check_walls(grid):
        build(grid)
        for port in grid.eigenmodeports:
            owner = port.owner
            masks = owner._cell_pec_electric_component_masks(grid)
            properties = (
                owner.complex_eps_r_uu,
                owner.complex_eps_r_vv,
                owner.complex_eps_r_ww,
            )
            # Carving a bore without averaging used to overwrite these live
            # Yee samples while the modal solver still imposed zero E on them.
            for mask, values in zip(masks, properties):
                assert not np.any(mask & np.isfinite(values))

    monkeypatch.setattr(FDTDGrid, "build", check_walls)
    monkeypatch.setattr(EigenmodeSource, "_should_plot_eigenmode_fields", lambda self: False)
    stem = tmp_path / "guide"
    gprMax.run(
        scenes=[module.build_scene(mode=mode)],
        outputfile=stem,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    with h5py.File(stem.with_suffix(".h5")) as output:
        source = output["eigenmode_ports/port1"]
        destination = output["eigenmode_ports/port2"]
        for port in (source, destination):
            assert np.all(port["power_wave_valid_S"])
            assert np.all(port["mode_tracking/numerical_valid"])
            assert port["mode_tracking/unresolved_interval_modes"].size == 0
            assert len(port.attrs["AnchorFrequencies"]) == len(module.ANCHORS)
        reflection_db = 20 * np.log10(np.abs(source["S"][mode - 1]))
        transmission_db = 20 * np.log10(np.abs(destination["S"][mode - 1]))
        record_property("maximum_s11_db", float(np.max(reflection_db)))
        record_property("maximum_s21_error_db", float(np.max(np.abs(transmission_db))))
        assert np.max(reflection_db) < -60
        assert np.max(np.abs(transmission_db)) < 0.005
        assert np.max(np.abs(source["S"][2 - mode])) < 1e-5
        assert np.max(np.abs(destination["S"][2 - mode])) < 1e-5

        neff = source["anchor_complex_neff"][...].real
        if example.startswith("example_8"):
            assert source["mode_tracking"].attrs["AutomaticDegenerateGroups"] == "1,2"
            np.testing.assert_allclose(neff[:, 0], neff[:, 1], atol=1e-10)
        else:
            difference = neff[:, 0] - neff[:, 1]
            assert difference[0] * difference[-1] < 0
            candidates = source["mode_tracking/candidate_indices"][...]
            assert candidates[0, 0] != candidates[-1, 0]
