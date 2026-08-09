"""Unit tests for ``gprMax/receivers.py``.

Conventions
-----------
* One behaviour per test; descriptive names following
  ``test_<unit>_<context>_<expected>``.
* Known bugs are pinned in dedicated tests with a clear docstring so a
  future fix that flips the assertion is obvious and intentional.

Out of scope (hardware-conditional, would need real GPU buffers):
    htod_rx_arrays cuda / opencl / metal real-device paths
    dtoh_rx_array Metal-specific buffer-copy path
"""

import sys
import types

import numpy as np
import pytest

from gprMax.receivers import Rx, dtoh_rx_array, htod_rx_arrays


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_rx(*, ID="rx0", x=0, y=0, z=0, outputs=None):
    rx = Rx()
    rx.ID = ID
    rx.xcoord = x
    rx.ycoord = y
    rx.zcoord = z
    if outputs is not None:
        rx.outputs = dict(outputs)
    return rx


class _FakeGpuArrayModule:
    """Stand-in for ``pycuda.gpuarray`` / ``pyopencl.array``.

    ``to_gpu`` and ``to_device`` return the host array untouched so
    tests can inspect what would have been shipped to the device.
    """

    @staticmethod
    def to_gpu(arr):
        return arr

    @staticmethod
    def to_device(_queue, arr):
        return arr


def _patch_pycuda(monkeypatch):
    fake_pycuda = types.ModuleType("pycuda")
    fake_gpuarray = _FakeGpuArrayModule()
    fake_pycuda.gpuarray = fake_gpuarray
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.gpuarray", fake_gpuarray)


# ---------------------------------------------------------------------------
# Rx — class-level constants
# ---------------------------------------------------------------------------


class TestRxAllowableOutputs:
    """The allowable-output lists are part of the public contract — the
    HDF5 writer and post-processing code keys off them.
    """

    def test_allowableoutputs_lists_nine_components_in_order(self):
        assert Rx.allowableoutputs == [
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "Ix",
            "Iy",
            "Iz",
        ]

    def test_defaultoutputs_is_first_six(self):
        """The default outputs are the E and H components; the trailing
        Ix/Iy/Iz currents only make sense for transmission-line sources.
        """
        assert Rx.defaultoutputs == ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    def test_allowableoutputs_dev_matches_defaultoutputs(self):
        """``allowableoutputs_dev`` and ``defaultoutputs`` are both the
        first six entries of ``allowableoutputs`` — currents are not
        supported on the GPU path. Reads like two names for one concept.
        """
        assert Rx.allowableoutputs_dev == Rx.defaultoutputs


# ---------------------------------------------------------------------------
# Rx — instance defaults
# ---------------------------------------------------------------------------


class TestRxDefaults:
    def test_outputs_starts_as_empty_dict(self):
        rx = Rx()
        assert rx.outputs == {}

    def test_coord_array_is_zero_int32(self):
        rx = Rx()
        assert rx.coord.dtype == np.int32
        assert rx.coord.shape == (3,)
        assert np.all(rx.coord == 0)

    def test_coordorigin_array_is_zero_int32(self):
        rx = Rx()
        assert rx.coordorigin.dtype == np.int32
        assert rx.coordorigin.shape == (3,)
        assert np.all(rx.coordorigin == 0)

    def test_outputs_is_per_instance_not_shared(self):
        """Sanity check: two ``Rx`` instances must not share the same
        outputs dict. A class-level mutable default would silently couple
        every receiver's recorded data.
        """
        a = Rx()
        b = Rx()
        a.outputs["Ex"] = np.zeros(5)
        assert "Ex" not in b.outputs


# ---------------------------------------------------------------------------
# Rx — coord properties
# ---------------------------------------------------------------------------


class TestRxCoordProperties:
    @pytest.mark.parametrize("axis, idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_coord_property_round_trips(self, axis, idx):
        rx = Rx()
        setattr(rx, f"{axis}coord", 12)
        assert rx.coord[idx] == 12
        assert getattr(rx, f"{axis}coord") == 12

    @pytest.mark.parametrize("axis, idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_coordorigin_property_round_trips(self, axis, idx):
        rx = Rx()
        setattr(rx, f"{axis}coordorigin", 4)
        assert rx.coordorigin[idx] == 4
        assert getattr(rx, f"{axis}coordorigin") == 4

    def test_coord_setter_preserves_int32_dtype(self):
        """The solver indexes 4D field arrays with these coords — losing
        the int32 dtype (e.g. by reassigning ``rx.coord = [...]``) would
        break the cython kernels. Setter writes into the existing array
        in place, so the dtype is preserved.
        """
        rx = Rx()
        rx.xcoord = 5
        assert rx.coord.dtype == np.int32


# ---------------------------------------------------------------------------
# htod_rx_arrays — shape, coord packing, missing CPU branch
# ---------------------------------------------------------------------------


class TestHtodRxArraysCpuBug:
    """Pin the missing CPU branch in ``htod_rx_arrays``.

    Source: ``receivers.py:90-140``. The function only assigns
    ``rxcoords_dev`` / ``rxs_dev`` inside ``cuda``/``opencl``/``metal``
    branches, so on CPU the final ``return`` accesses unbound locals.

    Mirrors the analogous bug in ``sources.htod_src_arrays``. When fixed
    (CPU branch added that returns the host numpy arrays), update this
    test to call ``htod_rx_arrays`` and assert the returned arrays have
    the expected shape.
    """

    def test_cpu_solver_raises_unbound_local(self, fake_grid):
        # solver defaults to "cpu" via the autouse receiver_config fixture.
        G = fake_grid(iterations=5, rxs=[])
        with pytest.raises(UnboundLocalError):
            htod_rx_arrays(G)


class TestHtodRxArraysCuda:
    """Verify shapes and coord packing through a fake ``pycuda.gpuarray``."""

    def test_returns_arrays_with_documented_shapes(self, fake_grid, monkeypatch):
        from gprMax import config

        config.sim_config.general["solver"] = "cuda"
        _patch_pycuda(monkeypatch)

        rxs = [_make_rx(x=1, y=2, z=3), _make_rx(x=4, y=5, z=6)]
        G = fake_grid(iterations=10, rxs=rxs)

        rxcoords_dev, rxs_dev = htod_rx_arrays(G)

        # rxcoords: (n_rxs, 3) of int32
        assert rxcoords_dev.shape == (2, 3)
        assert rxcoords_dev.dtype == np.int32
        # rxs: (allowableoutputs_dev=6, iterations, n_rxs)
        assert rxs_dev.shape == (len(Rx.allowableoutputs_dev), 10, 2)
        # Field-component array starts zeroed (solver fills it during the run).
        assert np.all(rxs_dev == 0)

    def test_packs_receiver_coords_in_declaration_order(
        self, fake_grid, monkeypatch
    ):
        from gprMax import config

        config.sim_config.general["solver"] = "cuda"
        _patch_pycuda(monkeypatch)

        rxs = [
            _make_rx(x=10, y=11, z=12),
            _make_rx(x=20, y=21, z=22),
            _make_rx(x=30, y=31, z=32),
        ]
        G = fake_grid(iterations=4, rxs=rxs)

        rxcoords_dev, _ = htod_rx_arrays(G)

        expected = np.array(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=np.int32
        )
        assert np.array_equal(rxcoords_dev, expected)


# ---------------------------------------------------------------------------
# dtoh_rx_array — non-Metal copy-back path
# ---------------------------------------------------------------------------


class TestDtohRxArrayHostPath:
    """The non-Metal branch at ``receivers.py:199-213`` walks every rx,
    finds the matching row in ``rxcoords_dev`` by coordinate equality,
    then copies each requested output's time series out of ``rxs_dev``.

    This branch currently only works correctly when both ``rxs_dev`` and
    ``rxcoords_dev`` are already host numpy arrays — the ``.get()`` calls
    needed to materialise CUDA/OpenCL gpuarrays are commented out
    (``receivers.py:200-201``). Tests here exercise the host-array case.
    """

    def test_copies_requested_outputs_into_rx_outputs(self, fake_grid):
        # Two receivers, both requesting Ex; iterations = 3.
        rx0 = _make_rx(x=1, y=2, z=3, outputs={"Ex": np.zeros(3)})
        rx1 = _make_rx(x=4, y=5, z=6, outputs={"Ex": np.zeros(3)})
        G = fake_grid(rxs=[rx0, rx1])

        rxcoords_dev = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        # rxs_dev shape: (n_outputs_dev=6, iterations=3, n_rxs=2)
        rxs_dev = np.zeros((len(Rx.allowableoutputs_dev), 3, 2))
        ex_idx = Rx.allowableoutputs_dev.index("Ex")
        rxs_dev[ex_idx, :, 0] = [1.0, 2.0, 3.0]   # rx0's Ex time series
        rxs_dev[ex_idx, :, 1] = [4.0, 5.0, 6.0]   # rx1's Ex time series

        dtoh_rx_array(rxs_dev, rxcoords_dev, G)

        assert np.array_equal(rx0.outputs["Ex"], [1.0, 2.0, 3.0])
        assert np.array_equal(rx1.outputs["Ex"], [4.0, 5.0, 6.0])

    def test_copies_multiple_outputs_for_one_rx(self, fake_grid):
        rx = _make_rx(
            x=0, y=0, z=0, outputs={"Ex": np.zeros(2), "Hy": np.zeros(2)}
        )
        G = fake_grid(rxs=[rx])
        rxcoords_dev = np.array([[0, 0, 0]], dtype=np.int32)
        rxs_dev = np.zeros((len(Rx.allowableoutputs_dev), 2, 1))
        rxs_dev[Rx.allowableoutputs_dev.index("Ex"), :, 0] = [0.1, 0.2]
        rxs_dev[Rx.allowableoutputs_dev.index("Hy"), :, 0] = [0.3, 0.4]

        dtoh_rx_array(rxs_dev, rxcoords_dev, G)

        assert np.array_equal(rx.outputs["Ex"], [0.1, 0.2])
        assert np.array_equal(rx.outputs["Hy"], [0.3, 0.4])

    def test_skips_rxs_whose_coords_do_not_match_any_row(self, fake_grid):
        """An rx whose coordinates don't appear in any row of
        ``rxcoords_dev`` should leave its outputs untouched.

        Uses ``len(G.rxs) == len(rxcoords_dev)`` to avoid the loop-bound
        bug (see ``TestDtohRxArrayLoopBoundBug``); the second row of
        ``rxcoords_dev`` contains coords that match neither rx.
        """
        rx_present = _make_rx(x=1, y=1, z=1, outputs={"Ex": np.zeros(2)})
        rx_missing = _make_rx(
            x=9, y=9, z=9, outputs={"Ex": np.full(2, fill_value=77.0)}
        )
        G = fake_grid(rxs=[rx_present, rx_missing])
        rxcoords_dev = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
        rxs_dev = np.zeros((len(Rx.allowableoutputs_dev), 2, 2))
        rxs_dev[Rx.allowableoutputs_dev.index("Ex"), :, 0] = [0.5, 0.6]

        dtoh_rx_array(rxs_dev, rxcoords_dev, G)

        assert np.array_equal(rx_present.outputs["Ex"], [0.5, 0.6])
        assert np.array_equal(rx_missing.outputs["Ex"], [77.0, 77.0])


class TestDtohRxArrayLoopBoundBug:
    """Pin the inner-loop bound bug at ``receivers.py:204``.

    The inner loop is ``for rxd in range(len(G.rxs))`` — it indexes
    ``rxcoords_dev[rxd, ...]`` using the receiver-list length. If a
    receiver is dropped (e.g. by MPI domain decomposition) so
    ``len(G.rxs) > len(rxcoords_dev)``, the line raises ``IndexError``.
    Correct fix would be ``range(len(rxcoords_dev))`` or a guard.

    When fixed, this test should assert the no-op behaviour instead.
    """

    def test_more_rxs_than_rxcoords_raises_indexerror(self, fake_grid):
        rx = _make_rx(x=0, y=0, z=0, outputs={"Ex": np.zeros(2)})
        ghost = _make_rx(x=9, y=9, z=9, outputs={"Ex": np.zeros(2)})
        G = fake_grid(rxs=[rx, ghost])
        rxcoords_dev = np.array([[0, 0, 0]], dtype=np.int32)  # only one row
        rxs_dev = np.zeros((len(Rx.allowableoutputs_dev), 2, 1))

        with pytest.raises(IndexError):
            dtoh_rx_array(rxs_dev, rxcoords_dev, G)
