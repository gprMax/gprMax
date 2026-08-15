from types import SimpleNamespace

import h5py
import numpy as np

import gprMax.fields_outputs as fields_outputs_mod
from gprMax.fields_outputs import (
    _global_position,
    _receiver_time_offset,
    _write_dual_lattice_source_excitation,
    _write_source_excitation,
    write_hdf5_outputfile,
)
from gprMax.sources import HertzianDipole, VoltageSource


def test_global_position_main_grid_scales_by_dl():
    """For the main grid, local index 0 coincides with the global origin,
    so _global_position should just scale the index by the discretisation.
    """
    grid = SimpleNamespace(dx=0.001, dy=0.002, dz=0.003)

    position = _global_position(grid, 10, 20, 30, is_subgrid=False)

    assert position == (0.01, 0.04, 0.09)


def test_global_position_subgrid_delegates_to_local_to_global():
    """For a subgrid, _global_position must go through
    SubGridBaseGrid.local_to_global rather than naively scaling the local
    index by dl - that naive scaling was the original bug (Position
    attribute ignored the subgrid's boundary/placement offset).
    """
    calls = []

    class FakeSubGrid:
        def local_to_global(self, coord):
            calls.append(coord)
            return (1.23, 4.56, 7.89)

    position = _global_position(FakeSubGrid(), 10, 20, 30, is_subgrid=True)

    assert calls == [(10, 20, 30)]
    assert position == (1.23, 4.56, 7.89)


def test_write_hdf5_outputfile_closes_file_via_context_manager(monkeypatch):
    """write_hdf5_outputfile() used to do `f = h5py.File(outputfile, "w")`
    with no with-block/close() - CPython's refcounting usually closes it
    once `f` goes out of scope, but an exception raised anywhere in the
    body (e.g. while writing subgrid data) can keep the frame - and so the
    file handle - alive via the traceback, leaking the descriptor or
    leaving the file incompletely flushed. Fixed by wrapping the whole
    function body in a `with h5py.File(...) as f:` block. This test
    confirms the file object is used as a context manager (i.e. actually
    closed), not just that .attrs assignments happen to work.
    """

    class _FakeAttrs(dict):
        pass

    class _FakeFile:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode
            self.attrs = _FakeAttrs()
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

        def create_group(self, path):
            raise AssertionError("no subgrids expected in this test")

    created = {}

    def _fake_h5py_file(filename, mode):
        f = _FakeFile(filename, mode)
        created["file"] = f
        return f

    monkeypatch.setattr(fields_outputs_mod.h5py, "File", _fake_h5py_file)
    monkeypatch.setattr(
        fields_outputs_mod, "write_hd5_data", lambda basegrp, grid, is_subgrid=False: None
    )

    model = SimpleNamespace(
        iterations=10,
        srcsteps=(0, 0, 0),
        rxsteps=(0, 0, 0),
        subgrids=[],
        G=SimpleNamespace(),
    )

    write_hdf5_outputfile(outputfile=SimpleNamespace(name="test.h5"), title="test", model=model)

    assert created["file"].entered
    assert created["file"].exited


def test_hertzian_source_excitation_preserves_half_step_samples(tmp_path):
    source = HertzianDipole()
    source.ID = "dipole"
    source.waveformID = "impulse"
    source.polarisation = "z"
    source.start = 0.0
    source.stop = 8e-9
    source.dl = 0.002
    source.waveformvalues_halfdt = np.asarray((1.0, 0.0, 0.0, 99.0), dtype=np.float32)
    grid = SimpleNamespace(
        iterations=3,
        dt=1e-12,
        waveforms=[SimpleNamespace(ID="impulse", type="impulse", amp=1.0, freq=1e9)],
    )
    output = tmp_path / "source.h5"

    with h5py.File(output, "w") as file:
        group = file.create_group("srcs/src1")
        _write_source_excitation(group, source, grid)

    with h5py.File(output, "r") as file:
        excitation = file["srcs/src1/excitation"]
        np.testing.assert_array_equal(excitation["samples"], (1.0, 0.0, 0.0))
        assert excitation.attrs["TimeSampleOffset"] == 0.5e-12
        assert excitation.attrs["DrivingQuantity"] == "electric_current"
        assert excitation.attrs["SpatialScale"] == 0.002


def test_hard_voltage_source_records_applied_electric_time(tmp_path):
    source = VoltageSource()
    source.ID = "hard"
    source.waveformID = "impulse"
    source.polarisation = "x"
    source.start = 0.0
    source.stop = 4e-9
    source.resistance = 0.0
    source.waveformvalues_wholedt = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    source.waveformvalues_halfdt = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    grid = SimpleNamespace(
        iterations=2,
        dt=2e-12,
        waveforms=[SimpleNamespace(ID="impulse", type="impulse", amp=1.0, freq=None)],
    )
    output = tmp_path / "source.h5"

    with h5py.File(output, "w") as file:
        group = file.create_group("srcs/src1")
        _write_source_excitation(group, source, grid)

    with h5py.File(output, "r") as file:
        excitation = file["srcs/src1/excitation"]
        assert excitation.attrs["TimeSampleOffset"] == grid.dt
        assert excitation.attrs["DrivingQuantity"] == "imposed_gap_voltage"


def test_receiver_time_offsets_follow_yee_staggering():
    dt = 3e-12
    assert _receiver_time_offset("Ez", dt) == 0.0
    assert _receiver_time_offset("Hy", dt) == -0.5 * dt
    assert _receiver_time_offset("Ix", dt) == -0.5 * dt


def test_transmission_line_excitation_exposes_whole_step_scalar_reference(tmp_path):
    source = SimpleNamespace(
        ID="line",
        waveformID="impulse",
        polarisation="z",
        start=0.0,
        stop=4e-9,
        waveformvalues_wholedt=np.asarray((1.0, 0.0, 0.0)),
        waveformvalues_halfdt=np.asarray((0.5, 0.0, 0.0)),
    )
    grid = SimpleNamespace(
        iterations=2,
        dt=2e-12,
        waveforms=[SimpleNamespace(ID="impulse", type="impulse", amp=1.0, freq=1.0)],
    )
    output = tmp_path / "line.h5"

    with h5py.File(output, "w") as file:
        group = file.create_group("tls/tl1")
        _write_dual_lattice_source_excitation(group, source, grid, "TransmissionLine")

    with h5py.File(output, "r") as file:
        excitation = file["tls/tl1/excitation"]
        assert excitation["samples"].id == excitation["samples_whole"].id
        assert excitation.attrs["TimeSampleOffset"] == 0.0
        assert excitation["samples_half"].attrs["TimeSampleOffset"] == grid.dt / 2
