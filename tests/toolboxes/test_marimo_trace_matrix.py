"""Tests for trace_matrix.py against synthetic single-trace HDF5 files."""

import h5py
import numpy as np
import pytest

from toolboxes.Marimo.h5_reader import load_file
from toolboxes.Marimo.trace_matrix import process_trace, stack_traces

DT = 4.717308673499368e-12
ITERATIONS = 100


def _write_trace(
    path,
    x_pos,
    amplitude=1.0,
    iterations=ITERATIONS,
    components=("Ez",),
    dt=DT,
):
    with h5py.File(path, "w") as f:
        f.attrs["Title"] = "synthetic"
        f.attrs["dt"] = dt
        f.attrs["dx_dy_dz"] = [0.002, 0.002, 0.002]
        f.attrs["Iterations"] = iterations
        f.attrs["nrx"] = 1
        f.attrs["nsrc"] = 1
        f.attrs["nx_ny_nz"] = [120, 105, 1]
        f.attrs["gprMax"] = "4.0.0"

        rx = f.create_group("rxs/rx1")
        rx.attrs["Name"] = "Rx(70,85,0)"
        rx.attrs["Position"] = [0.14, 0.17, 0.0]
        for comp in components:
            dataset = rx.create_dataset(
                comp, data=np.sin(np.linspace(0, 6.28, iterations)) * amplitude
            )
            dataset.attrs["SampleInterval"] = dt
            dataset.attrs["TimeSampleOffset"] = 0.0

        src = f.create_group("srcs/src1")
        src.attrs["Type"] = "HertzianDipole"
        src.attrs["Position"] = [x_pos, 0.17, 0.0]
    return path


def _write_trace_no_source(path, iterations=ITERATIONS):
    with h5py.File(path, "w") as f:
        f.attrs["Title"] = "no-source"
        f.attrs["dt"] = DT
        f.attrs["dx_dy_dz"] = [0.002, 0.002, 0.002]
        f.attrs["Iterations"] = iterations
        f.attrs["nrx"] = 1
        f.attrs["nsrc"] = 0
        f.attrs["nx_ny_nz"] = [120, 105, 1]
        f.attrs["gprMax"] = "4.0.0"
        rx = f.create_group("rxs/rx1")
        rx.attrs["Name"] = "Rx(70,85,0)"
        rx.attrs["Position"] = [0.14, 0.17, 0.0]
        rx.create_dataset("Ez", data=np.zeros(iterations))
    return path


def _write_trace_two_receivers(path, x_pos, iterations=ITERATIONS):
    with h5py.File(path, "w") as f:
        f.attrs["Title"] = "two-rx"
        f.attrs["dt"] = DT
        f.attrs["dx_dy_dz"] = [0.002, 0.002, 0.002]
        f.attrs["Iterations"] = iterations
        f.attrs["nrx"] = 2
        f.attrs["nsrc"] = 1
        f.attrs["nx_ny_nz"] = [120, 105, 1]
        f.attrs["gprMax"] = "4.0.0"

        rx1 = f.create_group("rxs/rx1")
        rx1.attrs["Name"] = "Rx(70,85,0)"
        rx1.attrs["Position"] = [0.14, 0.17, 0.0]
        rx1.create_dataset("Ez", data=np.ones(iterations) * 1.0)

        rx2 = f.create_group("rxs/rx2")
        rx2.attrs["Name"] = "Rx(90,85,0)"
        rx2.attrs["Position"] = [0.18, 0.17, 0.0]
        rx2.create_dataset("Ez", data=np.ones(iterations) * 2.0)

        src = f.create_group("srcs/src1")
        src.attrs["Type"] = "HertzianDipole"
        src.attrs["Position"] = [x_pos, 0.17, 0.0]
    return path


class TestProcessTrace:
    def test_success_extracts_component_and_position(self, tmp_path):
        fdata = load_file(_write_trace(tmp_path / "t1.h5", x_pos=0.05, amplitude=3.0))
        result = process_trace(fdata, "Ez", expected_len=None)
        assert result["ok"] is True
        assert result["component"] == "Ez"
        assert result["receiver"] == "rx1"
        assert result["x"] == pytest.approx(0.05)
        assert len(result["array"]) == ITERATIONS

    def test_falls_back_to_ez_when_preferred_missing(self, tmp_path):
        fdata = load_file(_write_trace(tmp_path / "t1.h5", x_pos=0.05))
        result = process_trace(fdata, "Hx", expected_len=None)
        assert result["ok"] is True
        assert result["component"] == "Ez"

    def test_falls_back_to_first_available_when_no_ez(self, tmp_path):
        fdata = load_file(_write_trace(tmp_path / "t1.h5", x_pos=0.05, components=("Hx", "Hy")))
        result = process_trace(fdata, "Ez", expected_len=None)
        assert result["ok"] is True
        assert result["component"] in ("Hx", "Hy")

    def test_mismatched_length_rejected(self, tmp_path):
        fdata = load_file(_write_trace(tmp_path / "t1.h5", x_pos=0.05, iterations=50))
        result = process_trace(fdata, "Ez", expected_len=100)
        assert result["ok"] is False
        assert "50 samples" in result["reason"]

    def test_missing_source_gives_none_position(self, tmp_path):
        fdata = load_file(_write_trace_no_source(tmp_path / "t1.h5"))
        result = process_trace(fdata, "Ez", expected_len=None)
        assert result["ok"] is True
        assert result["x"] is None

    def test_defaults_to_first_receiver_when_no_preference(self, tmp_path):
        fdata = load_file(_write_trace_two_receivers(tmp_path / "t1.h5", x_pos=0.05))
        result = process_trace(fdata, "Ez", expected_len=None)
        assert result["receiver"] == "rx1"
        assert result["array"][0] == pytest.approx(1.0)

    def test_honours_preferred_receiver(self, tmp_path):
        fdata = load_file(_write_trace_two_receivers(tmp_path / "t1.h5", x_pos=0.05))
        result = process_trace(fdata, "Ez", expected_len=None, preferred_receiver="rx2")
        assert result["receiver"] == "rx2"
        assert result["array"][0] == pytest.approx(2.0)

    def test_falls_back_to_first_receiver_if_preferred_not_present(self, tmp_path):
        fdata = load_file(_write_trace_two_receivers(tmp_path / "t1.h5", x_pos=0.05))
        result = process_trace(fdata, "Ez", expected_len=None, preferred_receiver="rx99")
        assert result["receiver"] == "rx1"


class TestStackTraces:
    def test_stacks_in_given_order(self, tmp_path):
        files = [
            load_file(_write_trace(tmp_path / f"t{i}.h5", x_pos=0.04 + i * 0.002, amplitude=i + 1))
            for i in range(5)
        ]
        result = stack_traces(files, preferred_component="Ez")
        assert result["matrix"].shape == (ITERATIONS, 5)
        assert result["positions_x"] == [pytest.approx(0.04 + i * 0.002) for i in range(5)]
        assert result["warnings"] == []
        assert result["all_positions_physical"] is True

    def test_amplitude_scaling_preserved_per_column(self, tmp_path):
        files = [
            load_file(_write_trace(tmp_path / f"t{i}.h5", x_pos=float(i), amplitude=i + 1))
            for i in range(4)
        ]
        result = stack_traces(files)
        peaks = [round(float(np.max(np.abs(result["matrix"][:, j])))) for j in range(4)]
        assert peaks == [1, 2, 3, 4]

    def test_empty_input_returns_none_matrix(self):
        result = stack_traces([])
        assert result["matrix"] is None
        assert result["positions_x"] == []
        assert result["warnings"] == []

    def test_mismatched_file_skipped_with_warning_rest_still_stacked(self, tmp_path):
        good = [
            load_file(_write_trace(tmp_path / "a.h5", x_pos=0.0)),
            load_file(_write_trace(tmp_path / "b.h5", x_pos=0.01)),
        ]
        bad = load_file(_write_trace(tmp_path / "c.h5", x_pos=0.02, iterations=50))
        result = stack_traces([good[0], bad, good[1]])
        assert result["matrix"].shape == (ITERATIONS, 2)
        assert len(result["warnings"]) == 1
        assert "trace 1" in result["warnings"][0]

    def test_equal_length_with_different_sample_times_is_skipped(self, tmp_path):
        first = load_file(_write_trace(tmp_path / "a.h5", x_pos=0.0))
        different_dt = load_file(_write_trace(tmp_path / "b.h5", x_pos=0.01, dt=1.1 * DT))
        result = stack_traces([first, different_dt])
        assert result["matrix"].shape == (ITERATIONS, 1)
        assert "sample times differ" in result["warnings"][0]

    def test_missing_source_falls_back_to_index_and_flags_non_physical(self, tmp_path):
        files = [
            load_file(_write_trace(tmp_path / "a.h5", x_pos=0.0)),
            load_file(_write_trace_no_source(tmp_path / "b.h5")),
        ]
        result = stack_traces(files)
        assert result["all_positions_physical"] is False
        assert result["positions_x"][1] == 1.0  # fell back to index

    def test_component_choice_is_consistent_across_stack(self, tmp_path):
        files = [
            load_file(_write_trace(tmp_path / f"t{i}.h5", x_pos=float(i), components=("Ez", "Hx")))
            for i in range(3)
        ]
        result = stack_traces(files, preferred_component="Hx")
        assert result["component"] == "Hx"
        assert result["matrix"].shape == (ITERATIONS, 3)

    def test_preferred_receiver_threads_through_stack(self, tmp_path):
        files = [
            load_file(_write_trace_two_receivers(tmp_path / f"t{i}.h5", x_pos=float(i)))
            for i in range(3)
        ]
        result = stack_traces(files, preferred_component="Ez", preferred_receiver="rx2")
        assert result["receiver"] == "rx2"
        assert np.all(result["matrix"] == 2.0)
