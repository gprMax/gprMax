"""Fit-plot lifecycle tests for microwave surface impedances."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.surface_impedance_plotting import surface_impedance_fit_plot_path
from gprMax.user_objects.cmds_multiuse import SurfaceImpedance
from testing.validation.impedance_surface._wall_waveguide_common import validation_cache_stem


@pytest.mark.parametrize(
    "geometry_only,plot_fit,expected",
    (
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (False, True, True),
    ),
)
def test_fit_plot_policy(monkeypatch, tmp_path, geometry_only, plot_fit, expected):
    output_stem = tmp_path / "model"
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(geometry_only=geometry_only))
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=output_stem),
    )
    grid = SimpleNamespace(surface_impedance_models={}, materials=[])
    surface = SurfaceImpedance(
        id="copper_wall",
        preset="copper",
        fit_frequency_range=(8e9, 12e9),
        plot_fit=plot_fit,
    )

    surface.build(grid)

    path = surface_impedance_fit_plot_path(output_stem, surface.ID)
    assert path.exists() is expected
    if expected:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000


def test_exact_resistance_has_no_fit_plot(monkeypatch, tmp_path):
    output_stem = tmp_path / "model"
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(geometry_only=True))
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=output_stem),
    )
    grid = SimpleNamespace(surface_impedance_models={}, materials=[])
    surface = SurfaceImpedance(id="resistive_wall", resistance=50)

    surface.build(grid)

    assert not surface_impedance_fit_plot_path(output_stem, surface.ID).exists()


def test_fit_plot_path_sanitises_model_id_without_losing_identity(tmp_path):
    path = surface_impedance_fit_plot_path(tmp_path / "model", "wall:section 1")
    assert path.parent == tmp_path
    assert "wall_section_1_" in path.name
    assert path.suffix == ".png"


def test_fit_plot_paths_distinguish_case_only_ids_on_windows(tmp_path):
    lower = surface_impedance_fit_plot_path(tmp_path / "model", "wall")
    upper = surface_impedance_fit_plot_path(tmp_path / "model", "WALL")

    assert lower.name.casefold() != upper.name.casefold()


def test_validation_cache_stem_covers_values_not_dictionary_order():
    first = validation_cache_stem(
        "wall",
        {"band": np.asarray((80e9, 100e9)), "time_window": np.float64(380e-12)},
    )
    equivalent = validation_cache_stem(
        "wall",
        {"time_window": 380e-12, "band": [80e9, 100e9]},
    )
    changed = validation_cache_stem(
        "wall",
        {"band": [80e9, 101e9], "time_window": 380e-12},
    )

    assert first == equivalent
    assert first != changed
