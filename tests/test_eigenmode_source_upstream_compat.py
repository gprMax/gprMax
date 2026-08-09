from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.sources as sources_module
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.materials import Material
from gprMax.sources import EigenmodeSource


def _materials():
    pec = Material(0, "pec")
    pec.se = float("inf")
    pmc = Material(1, "pmc")
    pmc.sm = float("inf")
    free_space = Material(2, "free_space")
    return pec, pmc, free_space


def _source(grid):
    source = EigenmodeSource(grid)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.transverse_start = np.array((0, 0), dtype=np.int32)
    source.transverse_stop = np.array((2, 2), dtype=np.int32)
    source.plane_index = 1
    source.frequency = 1e9
    return source


def test_modal_plot_policy_uses_geometry_only_default_and_explicit_overrides(
    monkeypatch,
):
    source = EigenmodeSource(None)
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(geometry_only=False),
    )

    source.plot_fields = None
    assert not source._should_plot_eigenmode_fields()
    config.sim_config.geometry_only = True
    assert source._should_plot_eigenmode_fields()
    source.plot_fields = False
    assert not source._should_plot_eigenmode_fields()
    source.plot_fields = True
    config.sim_config.geometry_only = False
    assert source._should_plot_eigenmode_fields()

    source.plot_waveform = None
    assert not source._should_plot_eigenmode_excitation()
    config.sim_config.geometry_only = True
    assert source._should_plot_eigenmode_excitation()
    source.plot_waveform = False
    assert not source._should_plot_eigenmode_excitation()
    source.plot_waveform = True
    config.sim_config.geometry_only = False
    assert source._should_plot_eigenmode_excitation()


def test_modal_plots_are_grouped_by_port_and_mode(monkeypatch, tmp_path):
    source = EigenmodeSource(None)
    source.plot_fields = True
    source.port_index = 2
    source.mode_count = 2
    source.mode_indices = (1, 2)
    source.frequency = 45e9
    source.frequencies = (45e9, 55e9, 65e9)
    source.mode_solver = object()
    source.mode_solvers = [object(), object(), object()]
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(input_file_path=tmp_path / "guide.in", geometry_only=False),
    )
    calls = []
    monkeypatch.setattr(
        sources_module,
        "plot_eigenmode_port_fields",
        lambda **kwargs: calls.append(kwargs),
    )

    source._plot_eigenmode_fields()

    assert [call["output_path"].name for call in calls] == [
        "guide_Port2_Mode1.png",
        "guide_Port2_Mode2.png",
    ]
    assert all(call["frequencies"] == (45e9, 55e9, 65e9) for call in calls)
    assert all(len(call["solvers"]) == 3 for call in calls)


@pytest.mark.parametrize(("plot_waveform", "expected_calls"), ((True, 1), (False, 0)))
def test_full_run_excitation_flag_controls_one_excitation_waveform_plot(
    monkeypatch,
    tmp_path,
    plot_waveform,
    expected_calls,
):
    source = EigenmodeSource(None)
    source.plot_fields = not plot_waveform
    source.plot_waveform = plot_waveform
    source.port_index = 2
    source.waveformID = "band_auto"
    source.spectral_threshold = 1e-3
    source.dft_start = 45e9
    source.dft_stop = 65e9
    source.start = 0.0
    source.broadband_input_waveform = np.arange(8, dtype=np.float64)
    source.port_monitor = SimpleNamespace(frequency=np.linspace(45e9, 65e9, 5))
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(input_file_path=tmp_path / "guide.in", geometry_only=False),
    )
    calls = []
    monkeypatch.setattr(
        sources_module,
        "plot_eigenmode_excitation",
        lambda **kwargs: calls.append(kwargs),
    )

    source._plot_eigenmode_excitation(SimpleNamespace(iterations=8, dt=1e-12))

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls[0]["output_path"].name == "guide_EigenmodeExcitation.png"
        np.testing.assert_array_equal(calls[0]["dft_frequencies"], source.port_monitor.frequency)


def test_unused_builtin_pmc_does_not_block_eigenmode_material_slice():
    """The built-in PMC may exist in the grid without being on the source plane."""
    pec, pmc, free_space = _materials()

    grid = SimpleNamespace(
        materials=[pec, pmc, free_space],
        ID=np.full((6, 3, 3, 3), free_space.numID, dtype=np.uint32),
    )
    source = _source(grid)

    tensors = source._extract_local_complex_property_tensors(grid, electric=False)

    assert [tensor.shape for tensor in tensors] == [(3, 2), (2, 3), (2, 2)]
    assert all(np.all(tensor == 1) for tensor in tensors)


def test_pmc_component_ids_become_infinite_permeability_on_source_slice():
    """Exact PMC H component positions are exposed to the solver as constraints."""
    pec, pmc, free_space = _materials()
    ids = np.full((6, 3, 3, 3), free_space.numID, dtype=np.uint32)
    ids[4, 1, 0, 0] = pmc.numID
    ids[5, 1, 0, 1] = pmc.numID
    ids[3, 1, 1, 1] = pmc.numID
    grid = SimpleNamespace(materials=[pec, pmc, free_space], ID=ids)
    source = _source(grid)

    mu_uu, mu_vv, mu_ww = source._extract_local_complex_property_tensors(grid, electric=False)

    assert np.isinf(mu_uu[0, 0])
    assert np.isinf(mu_vv[0, 1])
    assert np.isinf(mu_ww[1, 1])


def test_pmc_cell_expands_to_all_local_magnetic_faces():
    """A PMC cell constrains two own-axis H faces and its normal H face."""
    pec, pmc, free_space = _materials()
    grid = SimpleNamespace(
        materials=[pec, pmc, free_space],
        ID=np.full((6, 3, 3, 3), free_space.numID, dtype=np.uint32),
        solid=np.full((3, 3, 3), free_space.numID, dtype=np.uint32),
    )
    grid.solid[0, 0, 0] = pmc.numID
    source = _source(grid)

    pmc_u, pmc_v, pmc_w = source._cell_pmc_magnetic_component_masks(grid)

    expected_u = np.zeros((3, 2), dtype=bool)
    expected_u[0:2, 0] = True
    expected_v = np.zeros((2, 3), dtype=bool)
    expected_v[0, 0:2] = True
    expected_w = np.zeros((2, 2), dtype=bool)
    expected_w[0, 0] = True
    assert np.array_equal(pmc_u, expected_u)
    assert np.array_equal(pmc_v, expected_v)
    assert np.array_equal(pmc_w, expected_w)


def test_eigenmode_source_passes_cell_pmc_masks_to_solver(monkeypatch):
    """Cell-derived PMC masks reach the FDFD solver constructor."""
    pec, pmc, free_space = _materials()
    grid = SimpleNamespace(
        materials=[pec, pmc, free_space],
        ID=np.full((6, 3, 3, 3), free_space.numID, dtype=np.uint32),
        solid=np.full((3, 3, 3), free_space.numID, dtype=np.uint32),
        dl=np.full(3, 1e-3),
    )
    grid.solid[0, 0, 0] = pmc.numID
    source = _source(grid)
    source.mode_index = 1
    source.complex_eps_r_uu = np.ones((2, 3), dtype=np.complex128)
    source.complex_eps_r_vv = np.ones((3, 2), dtype=np.complex128)
    source.complex_eps_r_ww = np.ones((3, 3), dtype=np.complex128)
    source.complex_mu_r_uu = np.ones((3, 2), dtype=np.complex128)
    source.complex_mu_r_vv = np.ones((2, 3), dtype=np.complex128)
    source.complex_mu_r_ww = np.ones((2, 2), dtype=np.complex128)

    captured = {}

    class FakeSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_modes = 1
            self.Eu = np.zeros((2, 3, 1))
            self.Ev = np.zeros((3, 2, 1))
            self.Ew = np.zeros((3, 3, 1))
            self.Hu = np.zeros((3, 2, 1))
            self.Hv = np.zeros((2, 3, 1))
            self.Hw = np.zeros((2, 2, 1))
            self.complex_neff = np.asarray([1.0 + 0j])

        def solve(self):
            pass

    monkeypatch.setattr(sources_module, "FDFD_2D_mode_solver", FakeSolver)
    monkeypatch.setattr(source, "_plot_eigenmode_fields", lambda solver: None)
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float64}),
    )

    source._solve_eigenmode(grid)

    assert captured["pmc_u_mask"][0, 0]
    assert captured["pmc_u_mask"][1, 0]
    assert captured["pmc_v_mask"][0, 0]
    assert captured["pmc_v_mask"][0, 1]
    assert captured["pmc_w_mask"][0, 0]


def test_fdfd_solver_interprets_nonfinite_permeability_as_pmc(monkeypatch):
    """Direct component-sampled PMC values constrain H without explicit masks."""
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878188e-12,
                "m0": 1.25663706127e-6,
                "c": 299792458.0,
                "z0": 376.730313412,
            }
        ),
    )
    shape_cell = (2, 2)
    eps_uu = np.ones((2, 3), dtype=np.complex128)
    eps_vv = np.ones((3, 2), dtype=np.complex128)
    eps_ww = np.ones((3, 3), dtype=np.complex128)
    mu_uu = np.ones((3, 2), dtype=np.complex128)
    mu_vv = np.ones((2, 3), dtype=np.complex128)
    mu_ww = np.ones(shape_cell, dtype=np.complex128)
    mu_uu[1, 0] = np.inf
    mu_vv[0, 1] = np.inf
    mu_ww[1, 1] = np.inf

    solver = FDFD_2D_mode_solver(
        frequency=1e9,
        du=1e-3,
        dv=1e-3,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
    )

    assert solver.pmc_u_mask[1, 0]
    assert solver.pmc_v_mask[0, 1]
    assert solver.pmc_w_mask[1, 1]
    assert solver.ev_constraint_mask[1, 0]
    assert solver.eu_constraint_mask[0, 1]
    assert solver.mu_r_uu[1, 0] == 1
    assert solver.mu_r_vv[0, 1] == 1
    assert solver.mu_r_ww[1, 1] == 1
