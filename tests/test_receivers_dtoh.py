"""Regression test for gprMax/receivers.py's dtoh_rx_array() - Metal branch
never populated rx.outputs.

Found by an external Codex review: the Metal branch (solver == "metal")
read the GPU buffer into a local rxs_np array, but the loop that actually
copies values into rx.outputs[...] lived inside the `else:` block (the
CUDA/OpenCL path, where rxs_dev/rxcoords_dev arrive as host numpy arrays
already, converted by the caller via .get() before this function is
called). Since if/else are mutually exclusive, Metal receivers never got
their outputs written at all - they stayed whatever they were initialised
to (zero-filled), despite the buffer read itself succeeding.

Fixed by making the assignment loop run unconditionally after either
branch has produced host numpy arrays. Real Apple Metal hardware isn't
available in this environment, so this test exercises the exact "metal"
code path with a fake MTLBuffer-like stand-in exposing the same
.length()/.contents().as_buffer() interface dtoh_rx_array() calls, rather
than constructing a real Metal buffer.
"""
import numpy as np

import gprMax.config as config
from gprMax.receivers import Rx, dtoh_rx_array


class _FakeMetalBuffer:
    """Stand-in for a pyobjc MTLBuffer - exposes only the .length()/
    .contents().as_buffer() calls dtoh_rx_array()'s Metal branch uses."""

    def __init__(self, array: np.ndarray):
        self._bytes = array.tobytes()

    def length(self) -> int:
        return len(self._bytes)

    def contents(self):
        return self

    def as_buffer(self, size):
        return self._bytes


def test_metal_dtoh_rx_array_populates_rx_outputs(monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "metal"}
    config.sim_config.dtypes = {"float_or_double": np.float64}

    iterations = 4
    n_outputs = len(Rx.allowableoutputs_dev)

    rx = Rx()
    rx.xcoord, rx.ycoord, rx.zcoord = 1, 2, 3
    for output in Rx.defaultoutputs:
        rx.outputs[output] = np.zeros(iterations)

    class _DummyGrid:
        rxs = [rx]
        iterations = 4

    # rxs_shape: (field components, iterations, receivers) - fill with
    # values distinguishable by (component_index, iteration).
    rxs_shape = (n_outputs, iterations, 1)
    known = np.arange(np.prod(rxs_shape), dtype=np.float64).reshape(rxs_shape)
    fake_buffer = _FakeMetalBuffer(known)

    dtoh_rx_array(fake_buffer, rxcoords_dev=None, G=_DummyGrid())

    ex_index = Rx.allowableoutputs_dev.index("Ex")
    assert np.array_equal(rx.outputs["Ex"], known[ex_index, :, 0])
    assert not np.all(rx.outputs["Ex"] == 0), "Ex must not be left zero-filled"


def test_device_receiver_copy_uses_order_for_colocated_receivers(monkeypatch):
    """A hidden port receiver may share a coordinate with a public receiver."""

    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cuda"}
    config.sim_config.dtypes = {"float_or_double": np.float64}

    first = Rx()
    second = Rx()
    for rx in (first, second):
        rx.xcoord, rx.ycoord, rx.zcoord = 1, 2, 3
        rx.outputs["Ez"] = np.zeros(3)

    class _DummyGrid:
        rxs = [first, second]
        iterations = 3

    coordinates = np.asarray(((1, 2, 3), (1, 2, 3)), dtype=np.int32)
    values = np.zeros((len(Rx.allowableoutputs_dev), 3, 2), dtype=np.float64)
    ez = Rx.allowableoutputs_dev.index("Ez")
    values[ez, :, 0] = (1, 2, 3)
    values[ez, :, 1] = (10, 20, 30)

    dtoh_rx_array(values, coordinates, _DummyGrid())

    np.testing.assert_array_equal(first.outputs["Ez"], (1, 2, 3))
    np.testing.assert_array_equal(second.outputs["Ez"], (10, 20, 30))
