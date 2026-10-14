"""Undriven high-order tracking and native-field export regressions."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.solvers import Solver


def _example(name):
    path = (Path(__file__).resolve().parents[1] / 'examples/features/eigenmode_tracking' / f'{name}.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.parametrize('material', ('pec', 'copper', 'dielectric'))
@pytest.mark.parametrize('shape', ('rectangular', 'square', 'cylindrical'))
def test_sixteen_undriven_tracked_modes(tmp_path, monkeypatch, material, shape):
    def forbid_time_stepping(*args, **kwargs):
        pytest.fail('Mode inspection must not advance time')

    monkeypatch.setattr(Solver, 'solve', forbid_time_stepping)
    case = f'{material}_{shape}'
    scene = _example('gallery').build_scene(case, spacing=.0005)
    assert not any(isinstance(obj, (gprMax.EigenmodeExcitation, gprMax.VirtualWaveguide))
                   for obj in scene.grid_objects)
    gprMax.run(scenes=[scene], outputfile=tmp_path/case, geometry_only=True,
               cpu_precision='double', hide_progress_bars=True, log_level=40)
    with h5py.File(tmp_path/f'{case}.modes.h5') as output:
        port = output['ports/1']
        assert port['neff'].shape[0] == 5
        assert port['neff'].shape[1] >= 16
        assert np.all(np.isfinite(port['neff'][:]))
        assert np.all(port['anchor_mode_reference_valid'][:])
        assert np.all(port['anchor_mode_propagating'][:])
        assert np.all(port['mode_tracking/numerical_valid'][:])
        assert port['mode_tracking/unresolved_interval_modes'].size == 0
        if material == 'dielectric':
            assert np.all(port['neff'][:].real > 1)
        if material == 'pec':
            # Vacuum PEC fundamentals are below the air light line in this band.
            # Overwriting shared wall samples can create spurious surface modes.
            assert np.all(port['neff'][:].real < 1)
        for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            assert np.all(np.isfinite(port[f'fields/{component}/values'][:]))
    # Exercise plotting both with and without a degenerate symmetry block.
    if shape == 'rectangular':
        record = _example('plot_results').plot_case(tmp_path/f'{case}.modes.h5')
        assert record['mode_count'] >= 16
        assert all((tmp_path/name).is_file() for name in record['plots'])


@pytest.mark.integration
def test_frequency_crossing_example_geometry_only(tmp_path, monkeypatch):
    monkeypatch.setattr(Solver, 'solve', lambda *args, **kwargs: pytest.fail('Must not advance time'))
    scene = _example('example_10_anisotropic_frequency_crossing/tracked_modes').build_scene(spacing=.0005)
    assert not any(isinstance(obj, gprMax.EigenmodeExcitation) for obj in scene.grid_objects)
    gprMax.run(scenes=[scene], outputfile=tmp_path/'frequency_crossing', geometry_only=True,
               cpu_precision='double', hide_progress_bars=True, log_level=40)
    with h5py.File(tmp_path/'frequency_crossing.modes.h5') as output:
        port = output['ports/1']
        neff = port['neff'][:].real
        difference = neff[:, 0]-neff[:, 1]
        assert difference[0]*difference[-1] < 0
        candidates = port['mode_tracking/candidate_indices'][:]
        assert candidates[0, 0] != candidates[-1, 0]
        assert np.all(port['mode_tracking/numerical_valid'][:])


@pytest.mark.integration
def test_persistent_crossing_example_geometry_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(Solver, 'solve', lambda *args, **kwargs: pytest.fail('Must not advance time'))
    scene = _example('example_11_C4v_square_persistent_pair_crossing/tracked_modes').build_scene(2.3, 'before_crossing', spacing=.0005)
    assert not any(isinstance(obj, gprMax.EigenmodeExcitation) for obj in scene.grid_objects)
    gprMax.run(scenes=[scene], outputfile=tmp_path/'before_crossing', geometry_only=True,
               cpu_precision='double', hide_progress_bars=True, log_level=40)
    with h5py.File(tmp_path/'before_crossing.modes.h5') as output:
        port = output['ports/1']
        assert port['neff'].shape[1] == 8
        assert np.all(port['anchor_mode_reference_valid'][:])
        assert np.all(port['mode_tracking/numerical_valid'][:])
