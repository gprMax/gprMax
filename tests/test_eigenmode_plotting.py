from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from gprMax.eigenmode_plotting import (
    _source_dft,
    plot_eigenmode_excitation,
    plot_eigenmode_port_fields,
)
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver


def _full_vector_solver(frequency, scale=1.0):
    solver = object.__new__(FDFD_2D_mode_solver)
    solver.frequency = frequency
    solver.du = 0.2e-3
    solver.dv = 0.25e-3
    solver.Nu = 5
    solver.Nv = 4
    solver.num_modes = 1
    solver.complex_neff = np.asarray([1.25 - 2e-4j])

    u_edge = np.linspace(-1.0, 1.0, solver.Nu)[:, None]
    v_edge = np.linspace(-1.0, 1.0, solver.Nv)[None, :]
    solver.Eu = np.repeat((scale * u_edge)[:, :, None], solver.Nv + 1, axis=1)
    solver.Ev = np.repeat((-scale * v_edge)[:, :, None], solver.Nu + 1, axis=0)
    solver.Hu = np.repeat((0.002 * scale * v_edge)[:, :, None], solver.Nu + 1, axis=0)
    solver.Hv = np.repeat((0.002 * scale * u_edge)[:, :, None], solver.Nv + 1, axis=1)
    return solver


def test_vector_plot_uses_two_columns_and_one_row_per_anchor(tmp_path, monkeypatch):
    captured = {}
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        captured["figure"] = figure
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    output = tmp_path / "guide_Port1_Mode1.png"
    result = plot_eigenmode_port_fields(
        solvers=(_full_vector_solver(45e9), _full_vector_solver(65e9, scale=1.2)),
        frequencies=(45e9, 65e9),
        mode_index=1,
        port_index=1,
        output_path=output,
    )

    assert result == output
    assert output.is_file()
    main_axes = [axis for axis in captured["figure"].axes if axis.get_title()]
    assert len(main_axes) == 4
    assert ["Tangential E" in axis.get_title() for axis in main_axes].count(True) == 2
    assert ["Tangential H" in axis.get_title() for axis in main_axes].count(True) == 2
    assert sum("45 GHz" in axis.get_title() for axis in main_axes) == 2
    assert sum("65 GHz" in axis.get_title() for axis in main_axes) == 2


def test_vector_plot_rejects_mismatched_anchor_counts(tmp_path):
    output = Path(tmp_path) / "guide_Port1_Mode1.png"
    with pytest.raises(ValueError, match="counts must match"):
        plot_eigenmode_port_fields(
            solvers=(_full_vector_solver(45e9),),
            frequencies=(45e9, 65e9),
            mode_index=1,
            port_index=1,
            output_path=output,
        )


def test_source_dft_matches_direct_port_convention():
    dt = 2e-12
    samples = np.sin(2 * np.pi * 7e9 * np.arange(127) * dt)
    frequencies = np.linspace(5e9, 9e9, 9)
    times = np.arange(samples.size) * dt
    expected = dt * np.exp(-2j * np.pi * frequencies[:, None] * times) @ samples

    np.testing.assert_allclose(
        _source_dft(samples, dt, frequencies),
        expected,
        rtol=2e-12,
        atol=1e-25,
    )


def test_excitation_plot_has_waveform_spectrum_dft_bins_and_band(tmp_path, monkeypatch):
    captured = {}
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        captured["figure"] = figure
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    dt = 2e-12
    times = np.arange(256) * dt
    samples = np.exp(-((times - 0.25e-9) / 0.08e-9) ** 2) * np.cos(2 * np.pi * 7e9 * times)
    output = tmp_path / "guide_EigenmodeExcitation.png"
    result = plot_eigenmode_excitation(
        samples=samples,
        dt=dt,
        dft_frequencies=np.linspace(6e9, 8e9, 11),
        band_start=6e9,
        band_stop=8e9,
        port_index=1,
        waveform_id="band_auto",
        spectral_threshold=1e-3,
        output_path=output,
    )

    assert result == output
    assert output.is_file()
    assert len(captured["figure"].axes) == 2
    time_axis, spectrum_axis = captured["figure"].axes
    assert time_axis.get_title() == "Injected sampled waveform"
    assert spectrum_axis.get_title() == "Waveform DFT magnitude"
    assert len(spectrum_axis.collections) == 1
    assert spectrum_axis.patches
