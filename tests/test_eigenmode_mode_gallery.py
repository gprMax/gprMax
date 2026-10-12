"""Undriven high-order tracking and native-field export regressions."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.solvers import Solver


def _example(name):
    path = (Path(__file__).resolve().parents[1] / 'examples/features/eigenmode_ports'
            / 'example_10_tracked_mode_gallery' / f'{name}.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.parametrize('material', ('metallic', 'dielectric'))
@pytest.mark.parametrize('shape', ('rectangular', 'square', 'circular'))
def test_sixteen_undriven_tracked_modes(tmp_path, monkeypatch, material, shape):
    def forbid_time_stepping(*args, **kwargs):
        pytest.fail('Mode inspection must not advance time')

    monkeypatch.setattr(Solver, 'solve', forbid_time_stepping)
    case = f'{material}_{shape}'
    scene = _example('tracked_modes').build_scene(case)
    assert not any(isinstance(obj, (gprMax.EigenmodeExcitation, gprMax.VirtualWaveguide))
                   for obj in scene.grid_objects)
    gprMax.run(scenes=[scene], outputfile=tmp_path/case, geometry_only=True,
               cpu_precision='double', hide_progress_bars=True, log_level=40)
    with h5py.File(tmp_path/f'{case}.modes.h5') as output:
        port = output['ports/1']
        assert port['neff'].shape == (5, 16)
        assert np.all(np.isfinite(port['neff'][:]))
        assert np.all(port['anchor_mode_reference_valid'][:])
        assert np.all(port['anchor_mode_propagating'][:])
        assert np.all(port['mode_tracking/numerical_valid'][:])
        assert port['mode_tracking/unresolved_interval_modes'].size == 0
        if material == 'dielectric':
            assert np.all(port['neff'][:].real > 1)
        for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            assert np.all(np.isfinite(port[f'fields/{component}/values'][:]))
    # Exercise plotting both with and without a degenerate symmetry block.
    if shape == 'rectangular':
        record = _example('plot_results').plot_case(tmp_path/f'{case}.modes.h5')
        assert record['mode_count'] == 16
        assert all((tmp_path/name).is_file() for name in record['plots'])
