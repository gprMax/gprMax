from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.user_objects.cmds_multiuse as cmds_multiuse_module
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import EigenmodeRx, EigenmodeSource


def _configure_2d_grid(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(general={"solver": "cpu"}, mpi=False),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="2D TMz"),
    )
    grid = FDTDGrid()
    grid.size = np.asarray((100, 40, 1), dtype=np.int64)
    grid.dl = np.asarray((1e-3, 1e-3, 1e-3))
    grid.pmls["thickness"].update({"x0": 10, "xmax": 10, "y0": 5, "ymax": 5, "z0": 0, "zmax": 0})
    return grid


def _receiver(**overrides):
    kwargs = {
        "normal": "x",
        "direction": "-",
        "p1": (0.005, 0),
        "p2": (0.035, float("inf")),
        "w": 0.09,
        "mode_count": 2,
        "port_index": 2,
        "frequencies": (4e9, 6e9),
        "id": "output",
        "dft_start": 4e9,
        "dft_stop": 6e9,
        "dft_points": 21,
    }
    kwargs.update(overrides)
    return EigenmodeRx(**kwargs)


def test_eigenmode_rx_builds_multimode_port_at_pml_interface(monkeypatch):
    grid = _configure_2d_grid(monkeypatch)
    warnings = []
    monkeypatch.setattr(cmds_multiuse_module.logger, "warning", warnings.append)

    _receiver().build(grid)

    assert not warnings
    receiver = grid.eigenmodereceivers[0]
    assert receiver.mode_indices == (1, 2)
    assert receiver.mode_count == 2
    assert receiver.port_index == 2
    assert receiver.plane_index == 90
    assert receiver.dft_points == 21
    assert receiver.port_id == "output"


def test_eigenmode_rx_warns_away_from_pml(monkeypatch):
    grid = _configure_2d_grid(monkeypatch)
    warnings = []
    monkeypatch.setattr(cmds_multiuse_module.logger, "warning", warnings.append)

    _receiver(w=0.08).build(grid)

    assert any("not next to the xmax PML interface" in message for message in warnings)
    assert len(grid.eigenmodereceivers) == 1


def test_eigenmode_rx_warns_when_face_has_no_pml(monkeypatch):
    grid = _configure_2d_grid(monkeypatch)
    grid.pmls["thickness"]["xmax"] = 0
    warnings = []
    monkeypatch.setattr(cmds_multiuse_module.logger, "warning", warnings.append)

    _receiver(w=0.1).build(grid)

    assert any("not next to a PML" in message and "xmax face has zero PML thickness" in message for message in warnings)


def test_eigenmode_rx_rejects_zero_mode_count(monkeypatch):
    grid = _configure_2d_grid(monkeypatch)

    with pytest.raises(ValueError, match="one or greater"):
        _receiver(mode_count=0).build(grid)


def test_eigenmode_rx_mode_count_expands_to_consecutive_modes(monkeypatch):
    grid = _configure_2d_grid(monkeypatch)

    _receiver(mode_count=3).build(grid)

    assert grid.eigenmodereceivers[0].mode_indices == (1, 2, 3)


def test_hash_eigenmode_source_and_rx_parse_mode_counts_and_port_indices(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="2D TMz"),
    )
    commands = defaultdict(lambda: None)
    commands["#eigenmode_source"] = ["0.01 0.005 0 0.01 0.035 inf + 2,3 1 4e9 6e9 pulse 4e9 6e9 21"]
    commands["#eigenmode_rx"] = ["0.09 0.005 0 0.09 0.035 inf - 3 2 4e9 6e9 output 4e9 6e9 21"]

    objects = process_multicmds(commands)
    source = next(obj for obj in objects if isinstance(obj, EigenmodeSource))
    receiver = next(obj for obj in objects if isinstance(obj, EigenmodeRx))

    assert source.kwargs["mode_index"] == 2
    assert source.kwargs["mode_count"] == 3
    assert source.kwargs["port_index"] == 1
    assert receiver.kwargs["mode_count"] == 3
    assert receiver.kwargs["port_index"] == 2
    assert receiver.kwargs["frequencies"] == (4e9, 6e9)
    assert receiver.kwargs["dft_points"] == 21


def test_hash_eigenmode_source_single_mode_integer_defaults_mode_count(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="2D TMz"),
    )
    commands = defaultdict(lambda: None)
    commands["#eigenmode_source"] = ["0.01 0.005 0 0.01 0.035 inf + 2 1 5e9 pulse 5e9 5e9 1"]

    source = next(obj for obj in process_multicmds(commands) if isinstance(obj, EigenmodeSource))

    assert source.kwargs["mode_index"] == 2
    assert source.kwargs["mode_count"] == 2


@pytest.mark.parametrize(
    ("mode_token", "port_index", "message"),
    (
        ("0", 1, "excitation_mode must be one or greater"),
        ("2,1", 1, "mode_count must be at least excitation_mode"),
        ("1", 0, "port_index must be one or greater"),
    ),
)
def test_hash_eigenmode_source_rejects_invalid_mode_or_port(monkeypatch, mode_token, port_index, message):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="2D TMz"),
    )
    commands = defaultdict(lambda: None)
    commands["#eigenmode_source"] = [f"0.01 0.005 0 0.01 0.035 inf + {mode_token} {port_index} " "5e9 pulse 5e9 5e9 1"]

    with pytest.raises(ValueError, match=message):
        process_multicmds(commands)


def test_hash_eigenmode_rx_requires_exactly_one_source(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="2D TMz"),
    )
    commands = defaultdict(lambda: None)
    commands["#eigenmode_rx"] = ["0.09 0.005 0 0.09 0.035 inf - 2 2 5e9 output 5e9 5e9 1"]

    with pytest.raises(ValueError, match="one and only one"):
        process_multicmds(commands)


@pytest.mark.parametrize("source_count", (0, 2))
def test_grid_eigenmode_ports_require_exactly_one_source(source_count):
    grid = FDTDGrid()
    grid.eigenmodesources = [object() for _ in range(source_count)]
    grid.eigenmodereceivers = [object()]

    with pytest.raises(ValueError, match=rf"found {source_count}"):
        grid._eigenmode_port_grid_init()


def test_grid_eigenmode_ports_reject_duplicate_port_indices_before_solving():
    grid = FDTDGrid()
    grid.eigenmodesources = [SimpleNamespace(port_index=1)]
    grid.eigenmodereceivers = [SimpleNamespace(port_index=1)]

    with pytest.raises(ValueError, match="must be unique"):
        grid._eigenmode_port_grid_init()
