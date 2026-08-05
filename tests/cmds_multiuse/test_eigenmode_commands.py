from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import (
    EigenmodeBand,
    EigenmodeExcitation,
    EigenmodePort,
)


def _configure_grid(monkeypatch):
    monkeypatch.setattr(
        config,
        'sim_config',
        SimpleNamespace(general={'solver': 'cpu'}, mpi=False),
    )
    monkeypatch.setattr(config, 'get_model_config', lambda: SimpleNamespace(mode='2D TMz'))
    grid = FDTDGrid()
    grid.size = np.asarray((100, 40, 1), dtype=np.int64)
    grid.dl = np.asarray((1e-3, 1e-3, 1e-3))
    grid.dt = 1e-12
    grid.iterations = 4096
    grid.timewindow = grid.dt * grid.iterations
    grid.pmls['thickness'].update(
        {'x0': 10, 'xmax': 10, 'y0': 5, 'ymax': 5, 'z0': 0, 'zmax': 0}
    )
    return grid


def _build_definitions(grid, *, source_anchors='auto', receiver_anchors='auto'):
    EigenmodeBand(id='wg', fmin=4e9, fmax=6e9, points=21).build(grid)
    EigenmodePort(
        port=1,
        p1=(0.01, 0.005, 0),
        p2=(0.01, 0.035, float('inf')),
        direction='+',
        modes=(1, 2),
        anchors=source_anchors,
    ).build(grid)
    EigenmodePort(
        port=2,
        p1=(0.09, 0.005, 0),
        p2=(0.09, 0.035, float('inf')),
        direction='-',
        modes=(1,),
        anchors=receiver_anchors,
    ).build(grid)


def test_one_band_is_shared_while_each_port_owns_anchors(monkeypatch):
    grid = _configure_grid(monkeypatch)
    _build_definitions(grid, receiver_anchors=(2e9, 4e9, 6e9, 8e9))

    EigenmodeExcitation(
        port=1,
        mode=1,
        waveform='auto',
        plot_waveform=False,
    ).build(grid)

    assert grid.eigenmodeband.id == 'wg'
    assert grid.eigenmodeportdefs[1].anchor_policy == 'auto'
    assert grid.eigenmodeportdefs[2].anchor_policy == 'explicit'
    assert grid.eigenmodeportdefs[2].resolved_anchors == (2e9, 4e9, 6e9, 8e9)
    assert grid.eigenmodesources[0].mode_indices == (1, 2)
    assert grid.eigenmodereceivers[0].mode_indices == (1,)
    assert grid.eigenmodesources[0].dft_points == 21
    assert grid.eigenmodesources[0].plot_waveform is False
    assert grid.eigenmodereceivers[0].dft_points == 21
    assert grid.eigenmodeportdefs[1].resolved_anchors != grid.eigenmodeportdefs[2].resolved_anchors


def test_single_explicit_anchor_is_intentional_constant_basis(monkeypatch):
    grid = _configure_grid(monkeypatch)
    _build_definitions(grid, source_anchors=(5e9,), receiver_anchors=(5e9,))

    EigenmodeExcitation(port=1, mode=1, waveform='auto').build(grid)

    assert grid.eigenmodesources[0].frequencies == (5e9,)
    assert grid.eigenmodereceivers[0].frequencies == (5e9,)


def test_duplicate_global_band_is_rejected(monkeypatch):
    grid = _configure_grid(monkeypatch)
    EigenmodeBand(id='first', fmin=4e9, fmax=6e9, points=21).build(grid)

    with pytest.raises(ValueError, match='Exactly one EigenmodeBand'):
        EigenmodeBand(id='second', fmin=4e9, fmax=6e9, points=21).build(grid)


def test_hash_commands_parse_global_band_per_port_anchors_and_excitation(monkeypatch):
    monkeypatch.setattr(config, 'get_model_config', lambda: SimpleNamespace(mode='2D TMz'))
    commands = defaultdict(lambda: None)
    commands['#eigenmode_band'] = ['wg 4e9 6e9 21']
    commands['#eigenmode_port'] = [
        '1 0.01 0.005 0 0.01 0.035 inf + 1,2 auto y',
        '2 0.09 0.005 0 0.09 0.035 inf - 1 4e9 5e9 6e9 n',
    ]
    commands['#eigenmode_excitation'] = ['1 1 auto n']

    objects = process_multicmds(commands)

    band = next(obj for obj in objects if isinstance(obj, EigenmodeBand))
    ports = [obj for obj in objects if isinstance(obj, EigenmodePort)]
    excitation = next(obj for obj in objects if isinstance(obj, EigenmodeExcitation))
    assert band.kwargs == {'id': 'wg', 'fmin': 4e9, 'fmax': 6e9, 'points': 21}
    assert ports[0].kwargs['anchors'] == 'auto'
    assert ports[0].kwargs['plot_fields'] is True
    assert ports[1].kwargs['anchors'] == (4e9, 5e9, 6e9)
    assert ports[1].kwargs['plot_fields'] is False
    assert excitation.kwargs == {
        'port': 1,
        'mode': 1,
        'waveform': 'auto',
        'plot_waveform': False,
    }


def test_hash_excitation_plot_control_does_not_require_waveform_argument():
    commands = defaultdict(lambda: None)
    commands['#eigenmode_band'] = ['wg 4e9 6e9 21']
    commands['#eigenmode_port'] = ['1 0.01 0.005 0 0.01 0.035 inf + 1 auto']
    commands['#eigenmode_excitation'] = ['1 1 y']

    objects = process_multicmds(commands)

    excitation = next(obj for obj in objects if isinstance(obj, EigenmodeExcitation))
    assert excitation.kwargs == {'port': 1, 'mode': 1, 'plot_waveform': True}


def test_hash_ports_require_one_band_and_excitation():
    commands = defaultdict(lambda: None)
    commands['#eigenmode_port'] = ['1 0 0 0 0 1 1 + 1 auto']

    with pytest.raises(ValueError, match='exactly one #eigenmode_band'):
        process_multicmds(commands)


def test_grid_rejects_port_definitions_without_excitation():
    grid = FDTDGrid()
    grid.eigenmodeband = object()
    grid.eigenmodeportdefs[1] = object()

    with pytest.raises(ValueError, match='exactly one EigenmodeExcitation'):
        grid._eigenmode_port_grid_init()
