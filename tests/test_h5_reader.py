"""
tests/test_h5_reader.py: unit tests for toolboxes/Marimo/h5_reader.py

Run from repo root:
    pytest tests/test_h5_reader.py -v

Uses a synthetic HDF5 fixture built in memory — no real gprMax output
file required. Tests cover metadata extraction, component reading,
utility functions, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from toolboxes.Marimo.h5_reader import (
    build_label,
    format_metadata_text,
    get_time_axis,
    get_trace,
    get_unit_label,
    list_components,
    list_receivers,
    load_file,
    load_files,
)

# Fixtures


DT = 4.717308673499368e-12  # seconds (from real cylinder_Ascan_2D.h5)
ITERATIONS = 637
NRX = 1
NSRC = 1

COMPONENTS = {
    "Ex": np.zeros(ITERATIONS, dtype=np.float64),
    "Ey": np.zeros(ITERATIONS, dtype=np.float64),
    "Ez": np.linspace(-1000, 1000, ITERATIONS, dtype=np.float64),
    "Hx": np.ones(ITERATIONS, dtype=np.float64) * 0.5,
    "Hy": np.ones(ITERATIONS, dtype=np.float64) * -0.5,
    "Hz": np.zeros(ITERATIONS, dtype=np.float64),
}


def _write_synthetic_h5(path: Path, title: str = "Test A-scan") -> None:
    """Write a minimal valid gprMax v4 HDF5 file for testing."""
    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["Title"] = title
        f.attrs["dt"] = DT
        f.attrs["dx_dy_dz"] = np.array([0.002, 0.002, 0.002])
        f.attrs["Iterations"] = ITERATIONS
        f.attrs["nrx"] = NRX
        f.attrs["nsrc"] = NSRC
        f.attrs["nx_ny_nz"] = np.array([120, 105, 1])
        f.attrs["gprMax"] = "4.0.0b0"

        # Receiver
        rx1 = f.require_group("rxs/rx1")
        rx1.attrs["Name"] = "Rx(70,85,0)"
        rx1.attrs["Position"] = np.array([0.14, 0.17, 0.0])
        for comp, data in COMPONENTS.items():
            rx1.create_dataset(comp, data=data)

        # Source
        src1 = f.require_group("srcs/src1")
        src1.attrs["Type"] = "HertzianDipole"
        src1.attrs["Position"] = np.array([0.10, 0.17, 0.0])


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    """Single synthetic HDF5 file."""
    p = tmp_path / "test_ascan.h5"
    _write_synthetic_h5(p)
    return p


@pytest.fixture
def h5_file_b(tmp_path: Path) -> Path:
    """Second synthetic HDF5 file with a different title."""
    p = tmp_path / "test_ascan_freespace.h5"
    _write_synthetic_h5(p, title="Free space A-scan")
    return p


@pytest.fixture
def loaded(h5_file: Path):
    """Pre-loaded FileData dict."""
    return load_file(h5_file)


# load_file — metadata


class TestLoadFileMeta:
    def test_title(self, loaded):
        assert loaded["meta"]["title"] == "Test A-scan"

    def test_dt(self, loaded):
        assert loaded["meta"]["dt"] == pytest.approx(DT)

    def test_iterations(self, loaded):
        assert loaded["meta"]["iterations"] == ITERATIONS

    def test_nrx(self, loaded):
        assert loaded["meta"]["nrx"] == NRX

    def test_nsrc(self, loaded):
        assert loaded["meta"]["nsrc"] == NSRC

    def test_dx_dy_dz(self, loaded):
        assert loaded["meta"]["dx_dy_dz"] == pytest.approx([0.002, 0.002, 0.002])

    def test_nx_ny_nz(self, loaded):
        assert loaded["meta"]["nx_ny_nz"] == [120, 105, 1]

    def test_gprmax_version(self, loaded):
        assert loaded["meta"]["gprmax_version"] == "4.0.0b0"


# load_file — receiver structure


class TestLoadFileReceivers:
    def test_receiver_keys(self, loaded):
        assert "rx1" in loaded["receivers"]

    def test_receiver_name(self, loaded):
        assert loaded["receivers"]["rx1"]["name"] == "Rx(70,85,0)"

    def test_receiver_position(self, loaded):
        assert loaded["receivers"]["rx1"]["position"] == pytest.approx([0.14, 0.17, 0.0])

    def test_all_components_present(self, loaded):
        components = loaded["receivers"]["rx1"]["components"]
        for comp in COMPONENTS:
            assert comp in components

    def test_component_shape(self, loaded):
        ez = loaded["receivers"]["rx1"]["components"]["Ez"]
        assert ez.shape == (ITERATIONS,)

    def test_component_values(self, loaded):
        ez = loaded["receivers"]["rx1"]["components"]["Ez"]
        assert ez[0] == pytest.approx(COMPONENTS["Ez"][0])
        assert ez[-1] == pytest.approx(COMPONENTS["Ez"][-1])


# load_file — source structure


class TestLoadFileSources:
    def test_source_keys(self, loaded):
        assert "src1" in loaded["sources"]

    def test_source_type(self, loaded):
        assert loaded["sources"]["src1"]["type"] == "HertzianDipole"

    def test_source_position(self, loaded):
        assert loaded["sources"]["src1"]["position"] == pytest.approx([0.10, 0.17, 0.0])


# load_file — time axis


class TestLoadFileTimeAxis:
    def test_time_ns_length(self, loaded):
        assert len(loaded["time_ns"]) == ITERATIONS

    def test_time_ns_starts_at_zero(self, loaded):
        assert loaded["time_ns"][0] == pytest.approx(0.0)

    def test_time_ns_end_value(self, loaded):
        expected_end = (ITERATIONS - 1) * DT * 1e9
        assert loaded["time_ns"][-1] == pytest.approx(expected_end)


# load_file — error handling


class TestLoadFileErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent/path/file.h5")

    def test_invalid_hdf5(self, tmp_path):
        bad = tmp_path / "bad.h5"
        bad.write_text("not an hdf5 file")
        with pytest.raises(OSError):
            load_file(bad)


# load_files — multi-file


class TestLoadFiles:
    def test_returns_both_files(self, h5_file, h5_file_b):
        result = load_files([h5_file, h5_file_b])
        assert len(result) == 2

    def test_keyed_by_filename(self, h5_file):
        result = load_files([h5_file])
        assert "test_ascan.h5" in result

    def test_collision_uses_full_path(self, tmp_path):
        # Two files with identical names in different directories
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        p_a = dir_a / "ascan.h5"
        p_b = dir_b / "ascan.h5"
        _write_synthetic_h5(p_a)
        _write_synthetic_h5(p_b)
        result = load_files([p_a, p_b])
        # One entry will be the short name, one will be the full path
        assert len(result) == 2


# get_time_axis


class TestGetTimeAxis:
    def test_ns_unit(self, loaded):
        t = get_time_axis(loaded, unit="ns")
        assert t[-1] == pytest.approx((ITERATIONS - 1) * DT * 1e9)

    def test_s_unit(self, loaded):
        t = get_time_axis(loaded, unit="s")
        assert t[-1] == pytest.approx((ITERATIONS - 1) * DT)

    def test_iter_unit(self, loaded):
        t = get_time_axis(loaded, unit="iter")
        assert t[-1] == pytest.approx(ITERATIONS - 1)

    def test_invalid_unit(self, loaded):
        with pytest.raises(ValueError, match="Unknown unit"):
            get_time_axis(loaded, unit="ms")


# list_components


class TestListComponents:
    def test_returns_all_components(self, loaded):
        comps = list_components(loaded)
        assert set(comps) == set(COMPONENTS.keys())

    def test_sorted(self, loaded):
        comps = list_components(loaded)
        assert comps == sorted(comps)

    def test_missing_receiver_returns_empty(self, loaded):
        comps = list_components(loaded, receiver="rx99")
        assert comps == []


# list_receivers


class TestListReceivers:
    def test_returns_rx1(self, loaded):
        assert list_receivers(loaded) == ["rx1"]


# get_trace


class TestGetTrace:
    def test_ez_values(self, loaded):
        trace = get_trace(loaded, "Ez")
        np.testing.assert_array_almost_equal(trace, COMPONENTS["Ez"])

    def test_positive_polarity(self, loaded):
        trace = get_trace(loaded, "Hx", polarity=1)
        assert trace[0] == pytest.approx(COMPONENTS["Hx"][0])

    def test_negative_polarity_arg(self, loaded):
        trace = get_trace(loaded, "Hx", polarity=-1)
        assert trace[0] == pytest.approx(-COMPONENTS["Hx"][0])

    def test_negative_polarity_suffix(self, loaded):
        trace = get_trace(loaded, "Hx-")
        assert trace[0] == pytest.approx(-COMPONENTS["Hx"][0])

    def test_missing_component_raises(self, loaded):
        with pytest.raises(KeyError, match="Iz"):
            get_trace(loaded, "Iz")


# get_unit_label


class TestGetUnitLabel:
    def test_electric_field(self):
        assert get_unit_label("Ez") == "V/m"

    def test_magnetic_field(self):
        assert get_unit_label("Hx") == "A/m"

    def test_current(self):
        assert get_unit_label("Ix") == "A"

    def test_polarity_suffix_stripped(self):
        assert get_unit_label("Ez-") == "V/m"


# build_label


class TestBuildLabel:
    def test_label_format(self):
        label = build_label("cylinder_Ascan_2D.h5", "rx1", "Ez")
        assert label == "cylinder_Ascan_2D · rx1 · Ez"

    def test_stem_only(self):
        label = build_label("examples/cylinder_Ascan_2D.h5", "rx1", "Ez")
        assert label == "cylinder_Ascan_2D · rx1 · Ez"


# format_metadata_text


class TestFormatMetadataText:
    def test_contains_title(self, loaded):
        text = format_metadata_text(loaded)
        assert "Test A-scan" in text

    def test_contains_iterations(self, loaded):
        text = format_metadata_text(loaded)
        assert str(ITERATIONS) in text

    def test_contains_receiver_info(self, loaded):
        text = format_metadata_text(loaded)
        assert "rx1" in text
        assert "Rx(70,85,0)" in text
