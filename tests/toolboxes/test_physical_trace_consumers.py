"""Cross-consumer identity, physical time and native buffer-length checks."""

import ast
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

import gprMax
from gprMax.toolboxes.Marimo.h5_reader import get_trace, load_file
from gprMax.toolboxes.Marimo.reference import subtract_receiver_reference
from gprMax.toolboxes.Plotting import plot_Ascan, plot_Bscan
from gprMax.toolboxes.Plotting.plot_port import read_port_output
from gprMax.toolboxes.Utilities.outputfiles_merge import get_output_data, merge_files
from gprMax.toolboxes.Utilities.outputfiles_trace import collect_traces
from gprMax.toolboxes.Utilities.trace_time import read_time_history


def receiver_file(path, names=("A", "B"), dt=1e-10):
    with h5py.File(path, "w") as output:
        output.attrs.update(
            Iterations=8, dt=dt, nrx=len(names), nsrc=0, Title="trace tests", dx_dy_dz=[0.001] * 3, nx_ny_nz=[32] * 3
        )
        for index, name in enumerate(names, 1):
            group = output.create_group(f"rxs/rx{index}")
            group.attrs.update(Name=name, Position=[index, 0, 0])
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"):
                values = np.full(8, 10.0 if name == "A" else 2.0)
                dataset = group.create_dataset(component, data=values)
                dataset.attrs.update(SampleInterval=dt, TimeSampleOffset=0 if component.startswith("E") else -dt / 2)
    return path


def test_marimo_dashboard_subtraction_matches_identity(tmp_path):
    target = load_file(receiver_file(tmp_path / "target.h5"))
    background = load_file(receiver_file(tmp_path / "background.h5", ("B", "A")))
    # Execute the actual dashboard callback too, not just the reusable helper.
    module = ast.parse(Path("gprMax/toolboxes/Marimo/ascan_dashboard.py").read_text())
    helper = next(
        node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "_apply_subtraction"
    )
    namespace = dict(
        _ref_name="background",
        _files={"background": background},
        subtract_receiver_reference=subtract_receiver_reference,
        _sub_warnings=[],
        _subtracted=[],
    )
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "ascan_dashboard.py", "exec"), namespace)
    result = namespace["_apply_subtraction"](
        get_trace(target, "Ez", "rx1"), dict(filename="target", component="Ez", receiver="rx1", label="A"), target
    )
    np.testing.assert_array_equal(result, 0)
    assert not namespace["_sub_warnings"]
    assert namespace["_subtracted"] == ["A"]


@pytest.mark.parametrize("names", [("B",), ("A", "A")])
def test_background_missing_or_ambiguous_identity_is_rejected(tmp_path, names):
    target = load_file(receiver_file(tmp_path / "target.h5"))
    reference = load_file(receiver_file(tmp_path / "reference.h5", names))
    with pytest.raises(ValueError, match="identity"):
        subtract_receiver_reference(get_trace(target, "Ez"), target, reference, receiver="rx1", component="Ez")


def test_background_physical_times_checked_after_matching(tmp_path):
    target = load_file(receiver_file(tmp_path / "target.h5", dt=1e-15))
    path = receiver_file(tmp_path / "reference.h5", ("B", "A"), dt=1e-15)
    with h5py.File(path, "r+") as handle:
        handle["rxs/rx2/Ez"].attrs["TimeSampleOffset"] = 0.5e-15
    with pytest.raises(ValueError, match="sample times differ"):
        subtract_receiver_reference(get_trace(target, "Ez"), target, load_file(path), receiver="rx1", component="Ez")


@pytest.mark.parametrize(
    "outputs",
    [
        ("Ez", "Ix"),
        ("Ix", "Iy", "Iz"),
        ("Ex", "Hx", "Iz-"),
        ("Ix",),
        ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"),
    ],
)
def test_ascan_subsets_and_physical_axes(tmp_path, monkeypatch, outputs):
    path = receiver_file(tmp_path / "fields.h5", ("A",))
    captured = []
    monkeypatch.setattr(plot_Ascan, "handle_plot_output", lambda _plt, fig, *a, **k: captured.append(fig))
    try:
        plot_Ascan.mpl_plot(path, outputs=outputs, show=False)
        lines = [line for ax in captured[0].axes for line in ax.lines]
        assert len(lines) == len(outputs)
        for line in lines:
            label = line.get_label().strip("-")
            expected_offset = 0 if label.startswith("E") else -0.5e-10
            np.testing.assert_allclose(line.get_xdata(), expected_offset + np.arange(8) * 1e-10, rtol=1e-14, atol=0)
            if line.get_label().startswith("-"):
                np.testing.assert_array_equal(line.get_ydata(), -10)
    finally:
        plt.close("all")


def test_ascan_fft_uses_dataset_interval_and_offset(tmp_path, monkeypatch):
    path = receiver_file(tmp_path / "fields.h5", ("A",))
    with h5py.File(path, "r+") as handle:
        handle["rxs/rx1/Hx"].attrs.update(SampleInterval=2e-10, TimeSampleOffset=-1e-10)
    figures = []
    monkeypatch.setattr(plot_Ascan, "handle_plot_output", lambda _plt, fig, *a, **k: figures.append(fig))
    try:
        plot_Ascan.mpl_plot(path, outputs=["Hx"], fft=True, show=False)
        np.testing.assert_allclose(figures[0].axes[0].lines[0].get_xdata(), (np.arange(8) - 0.5) * 2e-10)
    finally:
        plt.close("all")


@pytest.mark.parametrize("component", ["Ez", "Hx", "Ix"])
def test_bscan_gather_preserves_sample_centres(tmp_path, monkeypatch, component):
    path = receiver_file(tmp_path / "fields.h5")
    samples, dt, offset = plot_Bscan.gather_receiver_outputs(path, component, return_time_offset=True)
    figures = []
    monkeypatch.setattr(plot_Bscan, "handle_plot_output", lambda _plt, fig, *a, **k: figures.append(fig))
    try:
        plot_Bscan.mpl_plot(path, samples, dt, 1, component, show=False, time_offset=offset)
        extent = figures[0].axes[0].images[0].get_extent()
        centres = extent[3] + (np.arange(8) + 0.5) * (extent[2] - extent[3]) / 8
        np.testing.assert_allclose(centres, offset + np.arange(8) * dt, atol=1e-25)
    finally:
        plt.close("all")


@pytest.mark.parametrize("attribute, value", [("TimeSampleOffset", 1e-10), ("SampleInterval", 2e-10)])
def test_gather_rejects_inconsistent_timing(tmp_path, attribute, value):
    path = receiver_file(tmp_path / "fields.h5")
    with h5py.File(path, "r+") as handle:
        handle["rxs/rx2/Ez"].attrs[attribute] = value
    with pytest.raises(ValueError, match="inconsistent sample times"):
        plot_Bscan.gather_receiver_outputs(path, "Ez")


def test_terminal_bscan_loader_preserves_offset_and_legacy_return(tmp_path, monkeypatch):
    path = tmp_path / "terminal.h5"
    dt, offset = 2e-10, 1e-10
    values = np.arange(16, dtype=float).reshape(8, 2)
    with h5py.File(path, "w") as output:
        output.attrs.update(Iterations=8, dt=dt, nrx=0)
        group = output.create_group("ports/feed")
        group.attrs["TimeSampleOffset"] = offset
        group["time"] = offset + np.arange(8) * dt
        group["Vtotal"] = values
    legacy = get_output_data(path, 1, "Vtotal", trace_group="ports/feed")
    assert len(legacy) == 2
    np.testing.assert_array_equal(legacy[0], values)
    samples, interval, start = get_output_data(path, 1, "Vtotal", trace_group="ports/feed", return_time_offset=True)
    assert interval == dt and start == offset
    figures = []
    monkeypatch.setattr(plot_Bscan, "handle_plot_output", lambda _plt, fig, *a, **k: figures.append(fig))
    try:
        plot_Bscan.mpl_plot(
            path, samples, interval, 1, "Vtotal", trace_group="ports/feed", time_offset=start, show=False
        )
        extent = figures[0].axes[0].images[0].get_extent()
        centres = extent[3] + (np.arange(8) + 0.5) * (extent[2] - extent[3]) / 8
        np.testing.assert_allclose(centres, offset + np.arange(8) * dt, atol=1e-25)
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "family, component, time_name, offset_attr, offset_steps",
    [
        ("tls", "Itotal", "time_current", "TimeCurrentOffset", -0.5),
        ("tls", "Vtotal", "time_voltage", "TimeVoltageOffset", 0),
        ("ports", "Vtotal", "time", "TimeSampleOffset", 0.5),
        ("ports", "Iloop", "time_current", "CurrentTimeSampleOffset", -0.5),
        ("ports", "Inetwork", "time", "TimeSampleOffset", 0.5),
        ("frills", "Itot", "time", "TimeOffset", 0),
    ],
)
@pytest.mark.parametrize("with_axis", [False, True])
def test_native_source_families_use_owning_grid_times(
    tmp_path, family, component, time_name, offset_attr, offset_steps, with_axis
):
    with h5py.File(tmp_path / "native.h5", "w") as output:
        output.attrs.update(Iterations=3, dt=3e-10)
        grid = output.create_group("subgrids/fine")
        grid.attrs.update(Iterations=9, dt=1e-10)
        group = grid.create_group(f"{family}/source")
        group.attrs[offset_attr] = offset_steps * 1e-10
        if with_axis:
            group[time_name] = (np.arange(9) + offset_steps) * 1e-10
        dataset = group.create_dataset(component, data=np.arange(10 if family == "frills" else 9, dtype=float))
        history = read_time_history(dataset)
        assert history.samples.size == 9
        assert history.dt == 1e-10
        np.testing.assert_allclose(history.time, (np.arange(9) + offset_steps) * 1e-10, atol=1e-25)


def test_reader_rejects_wrong_axis_without_inventing_replacement(tmp_path):
    with h5py.File(tmp_path / "bad.h5", "w") as output:
        output.attrs.update(Iterations=8, dt=1e-10)
        group = output.create_group("ports/feed")
        group["Vtotal"] = np.ones(8)
        group["time"] = np.arange(7) * 1e-10
        with pytest.raises(ValueError, match="Invalid time axis"):
            read_time_history(group["Vtotal"])
        del group["time"]
        group["time"] = np.arange(8) * 2e-10
        with pytest.raises(ValueError, match="disagree"):
            read_time_history(group["Vtotal"])


@pytest.fixture(scope="module")
def native_frill(tmp_path_factory):
    output = tmp_path_factory.mktemp("physical_frill") / "frill.h5"
    scene = gprMax.Scene()
    for item in (
        gprMax.Domain(p1=(0.02,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(iterations=64),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="w"),
        gprMax.Box(p1=(0, 0, 0), p2=(0.02, 0.02, 0.001), material_id="pec"),
        gprMax.ThinWire(p1=(0.01, 0.01, 0), p2=(0.01, 0.01, 0.01), radius=1e-4),
        gprMax.MagneticFrillSource(p1=(0.01, 0.01, 0), polarisation="z", zcoax=50, waveform_id="w"),
    ):
        scene.add(item)
    gprMax.run(scenes=[scene], outputfile=output, hide_progress_bars=True, log_level=50, cpu_precision="double")
    return output


@pytest.mark.integration
def test_real_frill_plot_export_and_merge_agree(native_frill, tmp_path):
    with h5py.File(native_frill) as output:
        physical_time = output["frills/frill1/time"][:]
        current = output["frills/frill1/Itot"][:64]
        assert output["frills/frill1/Itot"].size == 65
    records, dt, offset, *_ = collect_traces([native_frill], 1, "Itot", trace_group="frills/frill1")
    assert records[0].samples.size == 64 and offset == 0
    np.testing.assert_array_equal(records[0].samples, current)
    plot = read_port_output(native_frill, "frills/frill1")
    for trace in plot.time_traces:
        assert trace.values.size == 64
        np.testing.assert_allclose(trace.time, physical_time, atol=1e-25)
    voltage, _, voltage_offset = get_output_data(
        native_frill, 1, "Vtotal", trace_group="frills/frill1", return_time_offset=True
    )
    assert voltage.size == 64 and voltage_offset == 0
    merged = merge_files([native_frill, native_frill], tmp_path / "merged.h5")
    matrix, _, _ = get_output_data(merged, 1, "Vtotal", trace_group="frills/frill1", return_time_offset=True)
    assert matrix.shape == (64, 2)
    np.testing.assert_array_equal(matrix[:, 0], voltage)


@pytest.mark.integration
def test_real_frill_all_exporters_use_physical_length(native_frill, tmp_path):
    from gprMax.toolboxes.Utilities.outputfiles_seg2 import export_seg2
    from gprMax.toolboxes.Utilities.outputfiles_segy import export_segy
    from gprMax.toolboxes.Utilities.outputfiles_dt1 import export_dt1

    for function, suffix in ((export_seg2, "sg2"), (export_segy, "sgy"), (export_dt1, "dt1")):
        result = function([native_frill], tmp_path / f"frill.{suffix}", 1, "Itot", trace_group="frills/frill1")
        assert result.sample_count == 64
        assert result.time_sample_offset == 0
