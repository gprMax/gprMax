from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolboxes.Plotting.plot_port import (
    discover_port_outputs,
    main,
    plot_port_parameters,
    plot_port_signals,
    plot_port_validity,
    read_port_output,
    read_port_params,
    save_port_figures,
    select_port_paths,
)


def _primary_output(group, *, source_type="VoltageSource", offset=0):
    frequency = np.array([0, 0.5e9, 1e9, 1.5e9, 2e9])
    s11 = np.array([0.8, 0.4, 0.1 + 0.05j, 0.3, 0.6], dtype=np.complex128) + offset
    zin = 50 * (1 + s11) / (1 - s11)
    group.attrs["SourceType"] = source_type
    group.attrs["ReferenceImpedance"] = 50.0
    group.attrs["PortMode"] = "test"
    group.attrs["TailRelativeLevelDB"] = -55.0
    group.create_dataset("frequency", data=frequency)
    group.create_dataset("S11", data=s11)
    group.create_dataset("Zin", data=zin)
    group.create_dataset("Yin", data=1 / zin)
    group.create_dataset("valid_S11", data=[0, 1, 1, 1, 0])
    group.create_dataset("valid_Zin", data=[0, 1, 1, 1, 0])
    group.create_dataset("valid_Yin", data=[0, 1, 1, 1, 0])
    group.create_dataset("source_valid", data=[0, 1, 1, 1, 0])
    group.create_dataset("mesh_valid", data=[1, 1, 1, 1, 0])
    group.create_dataset("incident_relative_dB", data=[-100, -12, 0, -15, -45])
    group.create_dataset("cells_per_minimum_wavelength", data=[100, 40, 20, 13, 10])
    group.attrs["IncidentFloorDB"] = -40.0
    group.attrs["MinimumWavelengthCells"] = 10.0
    return frequency


def _write_voltage_port(output, path="ports/feed", offset=0):
    group = output.require_group(path)
    frequency = _primary_output(group, offset=offset)
    time = np.arange(8) * 1e-10
    group.create_dataset("time", data=time)
    group.create_dataset("Vgenerator", data=np.sin(np.arange(8)))
    group.create_dataset("Vtotal", data=0.8 * np.sin(np.arange(8)))
    group.create_dataset("Vincident_spectrum", data=np.ones(frequency.size, dtype=complex))
    group.create_dataset("Vtotal_spectrum", data=np.full(frequency.size, 0.8 + 0j))


def _write_network_port(output):
    group = output.require_group("ports/network")
    frequency = _primary_output(group, source_type="RationalNetworkTerminal")
    time = np.arange(8) * 1e-10
    group.create_dataset("time", data=time)
    group.create_dataset("Vgenerator", data=np.sin(np.arange(8)))
    group.create_dataset("Vtotal", data=0.7 * np.sin(np.arange(8)))
    group.create_dataset("Inetwork", data=0.01 * np.cos(np.arange(8)))
    group.create_dataset("Vgenerator_spectrum", data=np.ones(frequency.size, dtype=complex))
    group.create_dataset("Inetwork_spectrum", data=np.full(frequency.size, 0.01 + 0j))


def _write_transmission_line(output):
    group = output.require_group("tls/tl1")
    frequency = _primary_output(group, source_type="TransmissionLine")
    time = np.arange(8) * 1e-10
    group.create_dataset("time_voltage", data=time)
    group.create_dataset("time_current", data=time - 0.5e-10)
    for name, scale in (("Vinc", 1), ("Vtotal", 0.8), ("Iinc", 0.02), ("Itotal", 0.018)):
        group.create_dataset(name, data=scale * np.sin(np.arange(8)))
    group.create_dataset("Vincident_spectrum", data=np.ones(frequency.size, dtype=complex))
    group.create_dataset("Iincident_spectrum", data=np.full(frequency.size, 0.02 + 0j))
    group.create_dataset("Vreflected_current_spectrum", data=group["S11"][...])
    group.create_dataset("S11_current", data=group["S11"][...] * (1 + 1e-4j))
    group.create_dataset("Zin_current", data=group["Zin"][...] * (1 + 1e-4j))
    group.create_dataset("valid_S11_current", data=[0, 1, 1, 1, 0])
    group.create_dataset("valid_Zin_current", data=[0, 1, 1, 1, 0])
    group.create_dataset("line_propagation_valid", data=[1, 1, 1, 1, 0])


def _write_frill(output):
    group = output.require_group("frills/frill1")
    frequency = _primary_output(group, source_type="MagneticFrillSource")
    time = np.arange(8) * 1e-10
    group.create_dataset("time", data=time)
    group.create_dataset("Vinc", data=np.sin(np.arange(8)))
    group.create_dataset("Vtotal", data=0.6 * np.sin(np.arange(8)))
    group.create_dataset("Itot", data=0.012 * np.cos(np.arange(8)))
    group.create_dataset("Vincident_spectrum", data=np.ones(frequency.size, dtype=complex))
    group.create_dataset("Itotal_spectrum", data=np.full(frequency.size, 0.012 + 0j))


@pytest.fixture
def port_file(tmp_path):
    filename = tmp_path / "ports.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["dt"] = 1e-10
        _write_voltage_port(output)
        _write_network_port(output)
        _write_transmission_line(output)
        _write_frill(output)
        _write_voltage_port(output, "subgrids/fine/ports/subgrid_feed", offset=0.05)
    return filename


def test_discovery_and_repeated_selection_cover_every_current_port_location(port_file):
    available = discover_port_outputs(port_file)

    assert available == (
        "frills/frill1",
        "ports/feed",
        "ports/network",
        "subgrids/fine/ports/subgrid_feed",
        "tls/tl1",
    )
    assert select_port_paths(available, ("feed", "tls/tl1"), port_file) == (
        "ports/feed",
        "tls/tl1",
    )


def test_reader_uses_stored_primary_values_and_validity(port_file):
    port = read_port_output(port_file, "feed")

    assert port.path == "ports/feed"
    assert port.source_type == "VoltageSource"
    assert port.metadata["ReferenceImpedance"] == 50
    np.testing.assert_array_equal(port.primary_s11.valid, [False, True, True, True, False])
    assert [trace.name for trace in port.time_traces] == ["Vgenerator", "Vtotal"]
    assert [trace.name for trace in port.spectral_traces] == [
        "Vincident_spectrum",
        "Vtotal_spectrum",
    ]
    assert set(port.diagnostics) == {
        "incident_relative_dB",
        "cells_per_minimum_wavelength",
    }


def test_network_current_and_generator_spectrum_are_available(port_file):
    port = read_port_output(port_file, "network")

    assert "Inetwork" in [trace.name for trace in port.time_traces]
    assert "Vgenerator_spectrum" in [trace.name for trace in port.spectral_traces]
    assert "Inetwork_spectrum" in [trace.name for trace in port.spectral_traces]


def test_retired_reader_returns_a_safe_compatibility_view(port_file):
    result = read_port_params(port_file, "feed")

    np.testing.assert_array_equal(
        result["s11"], read_port_output(port_file, "feed").primary_s11.values
    )
    assert [trace["name"] for trace in result["time_traces"]["voltage"]] == [
        "Vgenerator",
        "Vtotal",
    ]
    assert result["time_traces"]["current"] == []


def test_transmission_line_diagnostics_are_preserved(port_file):
    port = read_port_output(port_file, "tl1")

    assert [trace.name for trace in port.s_parameters] == ["S11", "S11_current"]
    assert [trace.name for trace in port.impedances] == ["Zin", "Zin_current"]
    assert "line_propagation_valid" in port.validity_masks
    assert {trace.quantity for trace in port.time_traces} == {"voltage", "current"}
    assert "Vreflected_current_spectrum" in [trace.name for trace in port.spectral_traces]


def test_frill_voltage_and_current_histories_are_adaptive(port_file):
    port = read_port_output(port_file, "frill1")

    assert [trace.name for trace in port.time_traces] == ["Vinc", "Vtotal", "Itot"]
    assert {trace.quantity for trace in port.time_traces} == {"voltage", "current"}


def test_plots_save_with_port_specific_names_and_supported_formats(port_file, tmp_path):
    first = read_port_output(port_file, "feed")
    second = read_port_output(port_file, "tl1")
    first_figures = (
        plot_port_parameters(first),
        plot_port_signals(first),
        plot_port_validity(first),
    )
    second_figure = plot_port_parameters(second, show_invalid=True)

    first_paths = save_port_figures(
        port_file,
        first,
        output_dir=tmp_path,
        image_format="svg",
        parameters=first_figures[0],
        signals=first_figures[1],
        validity=first_figures[2],
    )
    second_paths = save_port_figures(
        port_file,
        second,
        output_dir=tmp_path,
        parameters=second_figure,
    )

    assert [path.name for path in first_paths] == [
        "ports_ports_feed_parameters.svg",
        "ports_ports_feed_signals.svg",
        "ports_ports_feed_validity.svg",
    ]
    assert second_paths[0].name == "ports_tls_tl1_parameters.png"
    assert all(path.is_file() and path.stat().st_size > 0 for path in first_paths + second_paths)
    for figure in (*first_figures, second_figure):
        plt.close(figure)


def test_cli_saves_all_ports_by_default(port_file, tmp_path):
    assert (
        main(
            [
                str(port_file),
                "--parameters-only",
                "--save",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert len(list(tmp_path.glob("*_parameters.png"))) == 5


def test_reader_rejects_wrong_frequency_shape(tmp_path):
    filename = Path(tmp_path) / "malformed.h5"
    with h5py.File(filename, "w") as output:
        group = output.require_group("ports/feed")
        group.create_dataset("frequency", data=[0, 1])
        group.create_dataset("S11", data=[1])

    with pytest.raises(ValueError, match="must be a one-dimensional array with 2 values"):
        read_port_output(filename)
