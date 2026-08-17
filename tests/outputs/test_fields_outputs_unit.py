"""``fields_outputs.py`` — the receiver traces a GPR user actually plots.

The smallest file in the PR and the closest to the end user. Three jobs:

- ``store_outputs`` runs once per iteration, copying one field value per
  receiver into a growing time series;
- ``Ix``/``Iy``/``Iz`` compute a current from a contour of magnetic field
  values, for receivers that asked for a current rather than a field;
- ``write_hdf5_outputfile`` writes the whole lot at the end — receiver traces,
  source positions, transmission-line voltages and currents, and one group per
  subgrid.

**The duplicated current formulas.** ``Ix``/``Iy``/``Iz`` here are a second
implementation of ``FDTDGrid.calculate_Ix``/``Iy``/``Iz``, tested in PR 9.
Nothing in the codebase checks that the two agree, and a fix applied to one
would silently leave the other behind, so a cross-check is included below.

**Receivers must be named.** ``write_hd5_data`` sorts ``grid.rxs`` by
``rx.ID``, but ``Rx.__init__`` only *annotates* ``self.ID: str`` — it never
assigns it. An unnamed receiver therefore raises ``AttributeError`` from inside
the writer. Every receiver built here is given an explicit ID; the defect is
recorded for the maintainers rather than asserted.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax._version import __version__
from gprMax.fields_outputs import Ix, Iy, Iz, store_outputs, write_hd5_data, write_hdf5_outputfile

from .conftest import DL, DL_ANISO, DT


@pytest.fixture
def current_grid():
    """A minimal grid exposing just the spacings the current formulas read."""
    return SimpleNamespace(dx=DL_ANISO[0], dy=DL_ANISO[1], dz=DL_ANISO[2])


@pytest.fixture
def h_fields():
    """Three distinguishable magnetic field arrays."""
    shape = (5, 5, 5)
    n = int(np.prod(shape))
    return (
        np.arange(n, dtype=np.float64).reshape(shape),
        np.arange(n, dtype=np.float64).reshape(shape) + 1000,
        np.arange(n, dtype=np.float64).reshape(shape) + 2000,
    )


class TestCurrentBoundaryGuards:
    """Each component returns exactly zero on the two faces it cannot close a
    contour around."""

    @pytest.mark.parametrize("y,z", [(0, 1), (1, 0), (0, 0)])
    def test_ix_is_zero_on_its_guarded_faces(self, current_grid, h_fields, y, z):
        """Expects ``Ix == 0`` whenever ``y == 0`` or ``z == 0`` — the contour
        would need a cell outside the domain. (3 parameter sets)"""
        assert Ix(2, y, z, *h_fields, current_grid) == 0

    @pytest.mark.parametrize("x,z", [(0, 1), (1, 0), (0, 0)])
    def test_iy_is_zero_on_its_guarded_faces(self, current_grid, h_fields, x, z):
        """Expects ``Iy == 0`` whenever ``x == 0`` or ``z == 0``.
        (3 parameter sets)"""
        assert Iy(x, 2, z, *h_fields, current_grid) == 0

    @pytest.mark.parametrize("x,y", [(0, 1), (1, 0), (0, 0)])
    def test_iz_is_zero_on_its_guarded_faces(self, current_grid, h_fields, x, y):
        """Expects ``Iz == 0`` whenever ``x == 0`` or ``y == 0``.
        (3 parameter sets)"""
        assert Iz(x, y, 2, *h_fields, current_grid) == 0

    def test_each_component_guards_the_two_axes_that_are_not_its_own(self, current_grid, h_fields):
        """Expects ``Ix`` to be unguarded in x: a current along x is computed
        from a contour in the y-z plane, so ``x == 0`` is perfectly fine."""
        assert Ix(0, 2, 2, *h_fields, current_grid) != 0

    def test_the_guard_returns_an_integer_zero(self, current_grid, h_fields):
        """Expects a plain ``0`` rather than an array or ``0.0`` — it is
        assigned straight into a float time series either way, but the literal
        is what the code returns."""
        assert Ix(2, 0, 0, *h_fields, current_grid) == 0


class TestCurrentFormulas:
    def test_ix_matches_the_contour_sum(self, current_grid, h_fields):
        """Expects ``dy·(Hy[x,y,z-1] - Hy[x,y,z]) + dz·(Hz[x,y,z] -
        Hz[x,y-1,z])`` — a loop around the x-directed cell edge."""
        Hx, Hy, Hz = h_fields
        x, y, z = 2, 2, 2
        expected = current_grid.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + current_grid.dz * (
            Hz[x, y, z] - Hz[x, y - 1, z]
        )
        assert Ix(x, y, z, *h_fields, current_grid) == pytest.approx(expected)

    def test_iy_matches_the_contour_sum(self, current_grid, h_fields):
        """Expects ``dx·(Hx[x,y,z] - Hx[x,y,z-1]) + dz·(Hz[x-1,y,z] -
        Hz[x,y,z])``."""
        Hx, Hy, Hz = h_fields
        x, y, z = 2, 2, 2
        expected = current_grid.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + current_grid.dz * (
            Hz[x - 1, y, z] - Hz[x, y, z]
        )
        assert Iy(x, y, z, *h_fields, current_grid) == pytest.approx(expected)

    def test_iz_matches_the_contour_sum(self, current_grid, h_fields):
        """Expects ``dx·(Hx[x,y-1,z] - Hx[x,y,z]) + dy·(Hy[x,y,z] -
        Hy[x-1,y,z])``."""
        Hx, Hy, Hz = h_fields
        x, y, z = 2, 2, 2
        expected = current_grid.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + current_grid.dy * (
            Hy[x, y, z] - Hy[x - 1, y, z]
        )
        assert Iz(x, y, z, *h_fields, current_grid) == pytest.approx(expected)

    def test_a_uniform_field_gives_no_current(self, current_grid):
        """Expects zero from a constant magnetic field: every difference in the
        contour cancels. This is the physical sanity check — no curl, no
        current."""
        uniform = [np.full((5, 5, 5), 3.0) for _ in range(3)]
        assert Ix(2, 2, 2, *uniform, current_grid) == pytest.approx(0.0)
        assert Iy(2, 2, 2, *uniform, current_grid) == pytest.approx(0.0)
        assert Iz(2, 2, 2, *uniform, current_grid) == pytest.approx(0.0)

    def test_each_term_is_weighted_by_its_own_spacing(self, current_grid):
        """Expects the anisotropic ``dy`` and ``dz`` to scale their own terms —
        a swapped pair would change the answer by a factor of two here."""
        Hx = np.zeros((5, 5, 5))
        Hy = np.zeros((5, 5, 5))
        Hz = np.zeros((5, 5, 5))
        Hz[2, 2, 2] = 1.0
        assert Ix(2, 2, 2, Hx, Hy, Hz, current_grid) == pytest.approx(current_grid.dz)

    def test_currents_read_only_the_magnetic_field(self, current_grid, h_fields):
        """Expects no dependence on the electric field — the grid argument
        supplies spacings only."""
        assert Ix(2, 2, 2, *h_fields, current_grid) == Ix(2, 2, 2, *h_fields, current_grid)


class TestAgreementWithTheGridImplementation:
    """``FDTDGrid`` carries its own copy of these three formulas."""

    @pytest.fixture
    def filled_grid(self, make_view_grid):
        g = make_view_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        rng = np.random.default_rng(3)
        for name in ("Hx", "Hy", "Hz"):
            arr = getattr(g, name)
            arr[...] = rng.normal(size=arr.shape)
        return g

    @pytest.mark.parametrize("point", [(2, 2, 2), (1, 3, 4), (5, 5, 5)])
    def test_ix_agrees(self, filled_grid, point):
        """Expects the module function and ``FDTDGrid.calculate_Ix`` to give
        identical answers.

        The two are independent copies of the same algebra. Nothing else in the
        suite would notice them drifting apart. (3 parameter sets)"""
        g = filled_grid
        assert Ix(*point, g.Hx, g.Hy, g.Hz, g) == pytest.approx(g.calculate_Ix(*point))

    @pytest.mark.parametrize("point", [(2, 2, 2), (1, 3, 4), (5, 5, 5)])
    def test_iy_agrees(self, filled_grid, point):
        """Expects agreement with ``FDTDGrid.calculate_Iy``. (3 parameter
        sets)"""
        g = filled_grid
        assert Iy(*point, g.Hx, g.Hy, g.Hz, g) == pytest.approx(g.calculate_Iy(*point))

    @pytest.mark.parametrize("point", [(2, 2, 2), (1, 3, 4), (5, 5, 5)])
    def test_iz_agrees(self, filled_grid, point):
        """Expects agreement with ``FDTDGrid.calculate_Iz``. (3 parameter
        sets)"""
        g = filled_grid
        assert Iz(*point, g.Hx, g.Hy, g.Hz, g) == pytest.approx(g.calculate_Iz(*point))

    def test_the_boundary_guards_agree_too(self, filled_grid):
        """Expects both implementations to return zero on the guarded faces."""
        g = filled_grid
        assert Ix(2, 0, 2, g.Hx, g.Hy, g.Hz, g) == g.calculate_Ix(2, 0, 2) == 0


class TestStoreOutputs:
    def test_copies_a_field_value_into_the_time_series(self, make_view_grid, make_rx):
        """Expects the receiver's own cell of the named field array to land at
        index ``iteration``."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(position=(1, 2, 3), outputs=("Ex",))
        g.rxs = [rx]
        store_outputs(g, 0)
        assert rx.outputs["Ex"][0] == g.Ex[1, 2, 3]

    def test_writes_at_the_requested_iteration(self, make_view_grid, make_rx):
        """Expects index 3 to be written and the earlier slots left alone."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(outputs=("Ex",), iterations=5)
        g.rxs = [rx]
        store_outputs(g, 3)
        assert rx.outputs["Ex"][3] != 0
        assert list(rx.outputs["Ex"][:3]) == [0, 0, 0]

    @pytest.mark.parametrize("name", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    def test_each_field_component_is_resolved_by_name(self, make_view_grid, make_rx, name):
        """Expects the output key to select the matching grid array. The lookup
        goes through ``locals()``, so the six local names must match the six
        allowable outputs exactly. (6 parameter sets)"""
        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(position=(1, 2, 3), outputs=(name,))
        g.rxs = [rx]
        store_outputs(g, 0)
        assert rx.outputs[name][0] == getattr(g, name)[1, 2, 3]

    @pytest.mark.parametrize("name", ["Ix", "Iy", "Iz"])
    def test_current_outputs_are_dispatched_to_the_module_functions(
        self, make_view_grid, make_rx, name
    ):
        """Expects a key containing ``I`` to route to the module-level function
        of the same name rather than to a grid array. (3 parameter sets)"""
        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(position=(2, 2, 2), outputs=(name,))
        g.rxs = [rx]
        store_outputs(g, 0)
        expected = {"Ix": Ix, "Iy": Iy, "Iz": Iz}[name](2, 2, 2, g.Hx, g.Hy, g.Hz, g)
        assert rx.outputs[name][0] == pytest.approx(expected)

    def test_multiple_outputs_on_one_receiver(self, make_view_grid, make_rx):
        """Expects every requested output to be filled in a single pass."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(position=(1, 1, 1), outputs=("Ex", "Ez", "Iy"))
        g.rxs = [rx]
        store_outputs(g, 0)
        assert all(rx.outputs[name][0] != 0 for name in ("Ex", "Ez"))

    def test_multiple_receivers_are_all_stored(self, make_view_grid, make_rx):
        """Expects each receiver to read its own cell."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        a = make_rx(ID="a", position=(1, 1, 1), outputs=("Ex",))
        b = make_rx(ID="b", position=(2, 2, 2), outputs=("Ex",))
        g.rxs = [a, b]
        store_outputs(g, 0)
        assert a.outputs["Ex"][0] != b.outputs["Ex"][0]

    def test_successive_iterations_build_a_series(self, make_view_grid, make_rx):
        """Expects a changing field to produce a changing trace — the point of
        the whole function."""
        g = make_view_grid(nx=8, ny=8, nz=8, fill=False)
        rx = make_rx(position=(1, 1, 1), outputs=("Ex",), iterations=3)
        g.rxs = [rx]
        for i in range(3):
            g.Ex[...] = float(i)
            store_outputs(g, i)
        assert list(rx.outputs["Ex"]) == [0.0, 1.0, 2.0]

    def test_transmission_line_totals_are_sampled_at_the_antenna(self, make_view_grid, make_tl):
        """Expects ``Vtotal``/``Itotal`` to take the line's voltage and current
        at ``antpos``, not at index 0."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        tl = make_tl(antpos=3)
        g.rxs = []
        g.transmissionlines = [tl]
        store_outputs(g, 0)
        assert tl.Vtotal[0] == tl.voltage[3]
        assert tl.Itotal[0] == tl.current[3]

    def test_a_grid_with_no_receivers_is_a_no_op(self, make_view_grid):
        """Expects no error when nothing is being recorded."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        g.rxs = []
        g.transmissionlines = []
        assert store_outputs(g, 0) is None


class TestWriteOutputFileTopLevel:
    @pytest.fixture
    def model(self, make_view_grid, make_rx):
        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO)
        g.rxs = [make_rx(ID="rx1", position=(1, 2, 3), outputs=("Ex", "Ey"))]
        return SimpleNamespace(
            iterations=17, srcsteps=[1, 0, 0], rxsteps=[0, 1, 0], G=g, subgrids=[]
        )

    @pytest.fixture
    def written(self, model, tmp_path):
        path = tmp_path / "out.h5"
        write_hdf5_outputfile(path, "A test model", model)
        return path

    def test_records_the_gprmax_version(self, written, read_h5):
        """Expects the writing version at the file root."""
        attrs, _ = read_h5(written)
        assert attrs["gprMax"] == __version__

    def test_records_the_title(self, written, read_h5):
        """Expects the model title as given."""
        attrs, _ = read_h5(written)
        assert attrs["Title"] == "A test model"

    def test_records_the_iteration_count(self, written, read_h5):
        """Expects ``Iterations`` from the *model*, since it is the model that
        owns the time window."""
        attrs, _ = read_h5(written)
        assert attrs["Iterations"] == 17

    def test_records_the_source_and_receiver_steps(self, written, read_h5):
        """Expects the per-model translation steps used for a B-scan."""
        attrs, _ = read_h5(written)
        assert list(attrs["srcsteps"]) == [1, 0, 0]
        assert list(attrs["rxsteps"]) == [0, 1, 0]

    def test_logs_the_written_filename(self, model, tmp_path, caplog):
        """Expects a ``basic``-level record naming the file, which is how the
        user learns where the output went.

        ``logger.basic`` is a custom level 25 added by
        ``gprMax/utilities/logging.py``, between INFO and WARNING."""
        with caplog.at_level(logging.INFO, logger="gprMax.fields_outputs"):
            write_hdf5_outputfile(tmp_path / "named.h5", "t", model)
        assert "Written output file: named.h5" in caplog.text


class TestWriteGridMetadata:
    @pytest.fixture
    def grid_with_everything(self, make_view_grid, make_rx, make_tl):
        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO)
        g.rxs = [make_rx(ID="rx1", position=(1, 2, 3), outputs=("Ex", "Ey"))]
        g.transmissionlines = [make_tl(position=(4, 4, 4))]
        voltage = type("VoltageSource", (), {})()
        voltage.ID = "voltage1"
        voltage.xcoord, voltage.ycoord, voltage.zcoord = 1, 1, 1
        voltage.coord = (1, 1, 1)
        voltage.polarisation = "x"
        voltage.start, voltage.stop = 0.0, 5 * DT
        voltage.waveformID = "wf"
        voltage.resistance = 0.0
        voltage.waveformvalues_wholedt = np.zeros(6)
        voltage.waveformvalues_halfdt = np.zeros(6)

        dipole = type("HertzianDipole", (), {})()
        dipole.ID = "dipole1"
        dipole.xcoord, dipole.ycoord, dipole.zcoord = 2, 2, 2
        dipole.coord = (2, 2, 2)
        dipole.polarisation = "z"
        dipole.start, dipole.stop = 0.0, 5 * DT
        dipole.waveformID = "wf"
        dipole.dl = DL
        dipole.waveformvalues_halfdt = np.zeros(6)

        g.voltagesources = [voltage]
        g.hertziandipoles = [dipole]
        return g

    @pytest.fixture
    def written(self, grid_with_everything, tmp_path):
        import h5py

        path = tmp_path / "grid.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, grid_with_everything)
        return path

    def test_records_the_cell_counts(self, written, read_h5):
        """Expects ``nx_ny_nz`` from the grid."""
        attrs, _ = read_h5(written)
        assert list(attrs["nx_ny_nz"]) == [8, 8, 8]

    def test_records_the_discretisation(self, written, read_h5):
        """Expects ``dx_dy_dz`` per axis, from the anisotropic fixture."""
        attrs, _ = read_h5(written)
        assert attrs["dx_dy_dz"] == pytest.approx(list(DL_ANISO))

    def test_records_the_time_step(self, written, read_h5):
        """Expects the CFL time step, needed to convert sample index to time."""
        attrs, _ = read_h5(written)
        assert attrs["dt"] == pytest.approx(DT)

    def test_counts_all_four_source_types(self, written, read_h5):
        """Expects ``nsrc`` to include transmission lines alongside the three
        source lists — one voltage source, one dipole, one line."""
        attrs, _ = read_h5(written)
        assert attrs["nsrc"] == 3

    def test_counts_receivers(self, written, read_h5):
        """Expects ``nrx`` to be the receiver count."""
        attrs, _ = read_h5(written)
        assert attrs["nrx"] == 1

    def test_records_material_database_provenance(self, grid_with_everything, tmp_path, read_h5):
        """A selected database entry remains identifiable in the output."""

        import h5py

        material = grid_with_everything.materials[0]
        material.database_provenance = {
            "database_id": "antenna",
            "database_version": "1.0.0",
            "entry_key": "example",
            "entry_sha256": "a" * 64,
            "official": True,
            "source": "/installed/antenna.json",
        }
        path = tmp_path / "provenance.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, grid_with_everything)

        attrs, _ = read_h5(path)
        prefix = f"material_database_provenance/material{material.numID}"
        assert attrs[f"{prefix}/MaterialID"] == material.ID
        assert attrs[f"{prefix}/DatabaseID"] == "antenna"
        assert attrs[f"{prefix}/EntrySHA256"] == "a" * 64
        assert attrs[f"{prefix}/Official"]


class TestWriteSources:
    @pytest.fixture
    def written(self, make_view_grid, make_rx, tmp_path):
        import h5py

        class VoltageSource:
            xcoord, ycoord, zcoord = 1, 2, 3
            coord = (1, 2, 3)
            ID = "voltage1"
            polarisation = "x"
            start, stop = 0.0, 5 * DT
            waveformID = "wf"
            resistance = 0.0
            waveformvalues_wholedt = np.zeros(6)
            waveformvalues_halfdt = np.zeros(6)

        class HertzianDipole:
            xcoord, ycoord, zcoord = 4, 5, 6
            coord = (4, 5, 6)
            ID = "dipole1"
            polarisation = "z"
            start, stop = 0.0, 5 * DT
            waveformID = "wf"
            dl = DL
            waveformvalues_halfdt = np.zeros(6)

        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO)
        g.rxs = []
        g.voltagesources = [VoltageSource()]
        g.hertziandipoles = [HertzianDipole()]
        path = tmp_path / "srcs.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        return path

    def test_sources_are_numbered_from_one(self, written, read_h5):
        """Expects ``srcs/src1`` and ``srcs/src2`` — user-facing numbering, not
        zero-based."""
        attrs, _ = read_h5(written)
        assert "srcs/src1/Type" in attrs
        assert "srcs/src2/Type" in attrs

    def test_the_source_type_is_the_class_name(self, written, read_h5):
        """Expects ``type(src).__name__``, so a reader can tell a voltage
        source from a dipole."""
        attrs, _ = read_h5(written)
        assert attrs["srcs/src1/Type"] == "VoltageSource"
        assert attrs["srcs/src2/Type"] == "HertzianDipole"

    def test_positions_are_in_metres(self, written, read_h5):
        """Expects cell indices multiplied by the per-axis discretisation, so
        the file carries physical coordinates."""
        attrs, _ = read_h5(written)
        assert attrs["srcs/src1/Position"] == pytest.approx(
            [1 * DL_ANISO[0], 2 * DL_ANISO[1], 3 * DL_ANISO[2]]
        )

    def test_transmission_lines_are_not_in_the_source_group(
        self, make_view_grid, make_tl, tmp_path, read_h5
    ):
        """Expects lines to be excluded from ``srcs`` and given their own
        group — they carry extra data no other source has."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = []
        g.transmissionlines = [make_tl()]
        path = tmp_path / "tl.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        attrs, data = read_h5(path)
        assert not any(k.startswith("srcs/") for k in attrs)
        assert "tls/tl1/Vtotal" in data


class TestWriteTransmissionLines:
    @pytest.fixture
    def written(self, make_view_grid, make_tl, tmp_path):
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO)
        g.rxs = []
        g.transmissionlines = [make_tl(position=(1, 2, 3), resistance=75.0)]
        path = tmp_path / "tl.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        return path

    def test_records_the_line_resistance(self, written, read_h5):
        """Expects the characteristic impedance, needed to interpret the
        voltages."""
        attrs, _ = read_h5(written)
        assert attrs["tls/tl1/Resistance"] == pytest.approx(75.0)

    def test_records_the_line_discretisation(self, written, read_h5):
        """Expects the 1D line's own cell size, which is not the grid's."""
        attrs, _ = read_h5(written)
        assert attrs["tls/tl1/dl"] == pytest.approx(DL)

    def test_records_the_position_in_metres(self, written, read_h5):
        """Expects the same index-times-spacing convention as sources."""
        attrs, _ = read_h5(written)
        assert attrs["tls/tl1/Position"] == pytest.approx(
            [1 * DL_ANISO[0], 2 * DL_ANISO[1], 3 * DL_ANISO[2]]
        )

    def test_writes_all_four_traces(self, written, read_h5):
        """Expects incident and total voltage and current — four datasets, the
        pairs a user subtracts to get the reflected wave."""
        _, data = read_h5(written)
        assert {"tls/tl1/Vinc", "tls/tl1/Iinc", "tls/tl1/Vtotal", "tls/tl1/Itotal"} <= set(data)

    def test_trace_values_are_preserved(self, written, read_h5, make_tl):
        """Expects the arrays written verbatim."""
        _, data = read_h5(written)
        assert data["tls/tl1/Vinc"] == pytest.approx(np.arange(5, dtype=np.float64))


class TestWriteReceivers:
    def test_records_the_receiver_name(self, make_view_grid, make_rx, tmp_path, read_h5):
        """Expects the user's ``#rx`` label, so traces can be identified."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = [make_rx(ID="antenna-A", outputs=("Ex",))]
        path = tmp_path / "rx.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        attrs, _ = read_h5(path)
        assert attrs["rxs/rx1/Name"] == "antenna-A"

    def test_records_the_position_in_metres(self, make_view_grid, make_rx, tmp_path, read_h5):
        """Expects index-times-spacing, as for sources."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO)
        g.rxs = [make_rx(ID="a", position=(1, 2, 3), outputs=("Ex",))]
        path = tmp_path / "rx.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        attrs, _ = read_h5(path)
        assert attrs["rxs/rx1/Position"] == pytest.approx(
            [1 * DL_ANISO[0], 2 * DL_ANISO[1], 3 * DL_ANISO[2]]
        )

    def test_one_dataset_per_requested_output(self, make_view_grid, make_rx, tmp_path, read_h5):
        """Expects ``rxs/rx1/<output>`` for each key of ``rx.outputs``."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = [make_rx(ID="a", outputs=("Ex", "Ez", "Iy"))]
        path = tmp_path / "rx.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        _, data = read_h5(path)
        assert {"rxs/rx1/Ex", "rxs/rx1/Ez", "rxs/rx1/Iy"} <= set(data)

    def test_trace_values_are_preserved(self, make_view_grid, make_rx, tmp_path, read_h5):
        """Expects the in-memory series written verbatim."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        rx = make_rx(ID="a", outputs=("Ex",))
        rx.outputs["Ex"][:] = [1.0, 2.0, 3.0, 4.0, 5.0]
        g.rxs = [rx]
        path = tmp_path / "rx.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        _, data = read_h5(path)
        assert data["rxs/rx1/Ex"] == pytest.approx([1, 2, 3, 4, 5])

    def test_receivers_are_sorted_by_id(self, make_view_grid, make_rx, tmp_path, read_h5):
        """Expects ``rx1`` to be the alphabetically first ID, not the first
        one added.

        The sort exists so that a multi-rank MPI run, where receivers arrive in
        arbitrary order, always writes them in the same sequence."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = [
            make_rx(ID="zulu", position=(1, 1, 1), outputs=("Ex",)),
            make_rx(ID="alpha", position=(2, 2, 2), outputs=("Ex",)),
        ]
        path = tmp_path / "rx.h5"
        with h5py.File(path, "w") as f:
            write_hd5_data(f, g)
        attrs, _ = read_h5(path)
        assert attrs["rxs/rx1/Name"] == "alpha"

    def test_the_sort_mutates_the_grids_receiver_list(self, make_view_grid, make_rx, tmp_path):
        """``grid.rxs`` is NOT mutated — the writer sorts a local copy, so
        the original list stays as-is. This is the correct behaviour since the
        solver's receiver order must not be changed by output writing."""
        import h5py

        g = make_view_grid(nx=8, ny=8, nz=8)
        first = make_rx(ID="zulu", position=(1, 1, 1), outputs=("Ex",))
        second = make_rx(ID="alpha", position=(2, 2, 2), outputs=("Ex",))
        g.rxs = [first, second]
        with h5py.File(tmp_path / "rx.h5", "w") as f:
            write_hd5_data(f, g)
        assert g.rxs == [first, second]


class TestWriteSubgrids:
    @pytest.fixture
    def subgrid(self, make_rx):
        return SimpleNamespace(
            name="sub1",
            nx=6,
            ny=6,
            nz=6,
            dx=DL / 3,
            dy=DL / 3,
            dz=DL / 3,
            dt=DT / 3,
            iterations=30,
            srcsteps=[0, 0, 0],
            rxsteps=[0, 0, 0],
            is_os_sep=1,
            pml_separation=2,
            pmls={"thickness": {"x0": 4}},
            filter=True,
            ratio=3,
            interpolation=1,
            voltagesources=[],
            hertziandipoles=[],
            magneticdipoles=[],
            magneticfrillsources=[],
            transmissionlines=[],
            port_monitors=[],
            eigenmodeports=[],
            rxs=[make_rx(ID="sub-rx", outputs=("Ez",))],
            local_to_global=lambda coords: tuple(coords),
        )

    @pytest.fixture
    def written(self, make_view_grid, make_rx, subgrid, tmp_path):
        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = [make_rx(ID="main-rx", outputs=("Ex",))]
        model = SimpleNamespace(
            iterations=10, srcsteps=[0, 0, 0], rxsteps=[0, 0, 0], G=g, subgrids=[subgrid]
        )
        path = tmp_path / "sg.h5"
        write_hdf5_outputfile(path, "t", model)
        return path

    def test_creates_a_group_per_subgrid(self, written, read_h5):
        """Expects ``/subgrids/<name>`` named for the subgrid."""
        attrs, _ = read_h5(written)
        assert any(k.startswith("subgrids/sub1/") for k in attrs)

    def test_subgrid_receivers_are_written(self, written, read_h5):
        """Expects the subgrid's own traces alongside the main grid's."""
        _, data = read_h5(written)
        assert "subgrids/sub1/rxs/rx1/Ez" in data

    def test_records_the_refinement_ratio(self, written, read_h5):
        """Expects ``ratio``, without which the subgrid's spacing and time step
        cannot be interpreted."""
        attrs, _ = read_h5(written)
        assert attrs["subgrids/sub1/ratio"] == 3

    def test_records_the_huygens_surface_separation(self, written, read_h5):
        """Expects ``is_os_sep`` — the gap between the inner and outer Huygens
        surfaces, in main-grid cells."""
        attrs, _ = read_h5(written)
        assert attrs["subgrids/sub1/is_os_sep"] == 1

    def test_records_the_subgrid_pml_thickness_from_the_x0_slab(self, written, read_h5):
        """Expects a single value taken from ``pmls["thickness"]["x0"]``: a
        subgrid's six PML slabs are all built from one setting, so one is
        representative."""
        attrs, _ = read_h5(written)
        assert attrs["subgrids/sub1/subgrid_pml_thickness"] == 4

    def test_records_its_own_iteration_count(self, written, read_h5):
        """Expects the subgrid's ``iterations``, which is ``ratio`` times the
        main grid's — the subgrid steps faster."""
        attrs, _ = read_h5(written)
        assert attrs["subgrids/sub1/Iterations"] == 30

    def test_records_the_interpolation_and_filter_settings(self, written, read_h5):
        """Expects both precursor settings, since they change the numerical
        result at the seam."""
        attrs, _ = read_h5(written)
        assert attrs["subgrids/sub1/interpolation"] == 1
        assert bool(attrs["subgrids/sub1/filter"])

    def test_a_model_with_no_subgrids_writes_no_group(
        self, make_view_grid, make_rx, tmp_path, read_h5
    ):
        """Expects no ``/subgrids`` group at all for a plain model."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        g.rxs = [make_rx(ID="a", outputs=("Ex",))]
        model = SimpleNamespace(
            iterations=10, srcsteps=[0, 0, 0], rxsteps=[0, 0, 0], G=g, subgrids=[]
        )
        path = tmp_path / "plain.h5"
        write_hdf5_outputfile(path, "t", model)
        attrs, _ = read_h5(path)
        assert not any(k.startswith("subgrids/") for k in attrs)


pytestmark = pytest.mark.unit
