"""Device-data and shared-template tests for transmission-line sources."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.cuda_opencl import knl_transmission_line
from gprMax.sources import (
    TransmissionLine,
    dtoh_transmission_line_outputs,
    transmission_line_host_arrays,
)


def _line(polarisation="z", coord=(2, 3, 4)):
    iterations = 4
    return SimpleNamespace(
        ID="tl",
        xcoord=coord[0],
        ycoord=coord[1],
        zcoord=coord[2],
        polarisation=polarisation,
        nl=3,
        srcpos=1,
        antpos=2,
        start=0.15,
        stop=0.30,
        resistance=50.0,
        abcv0=0.25,
        abcv1=-0.5,
        voltage=np.array([1.0, 2.0, 3.0]),
        current=np.array([4.0, 5.0, 6.0]),
        waveformvalues_wholedt=np.arange(iterations + 1, dtype=np.float64),
        waveformvalues_halfdt=np.arange(iterations + 1, dtype=np.float64) + 0.5,
        Vtotal=np.zeros(iterations + 1, dtype=np.float64),
        Itotal=np.zeros(iterations + 1, dtype=np.float64),
    )


@pytest.fixture
def float64_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float64}),
    )


def test_transmission_line_host_arrays_pack_state_and_activity(float64_config):
    line = _line()
    grid = SimpleNamespace(iterations=4, dt=0.1)

    arrays = transmission_line_host_arrays([line], grid)

    # The host reproduces the CPU source's direct iteration*dt comparison;
    # 3*0.1 is slightly greater than the requested stop time 0.3.
    assert arrays["info"].tolist() == [[2, 3, 4, 2, 0, 3, 1, 2, 2, 2]]
    np.testing.assert_array_equal(arrays["voltage"], line.voltage)
    np.testing.assert_array_equal(arrays["current"], line.current)
    np.testing.assert_array_equal(arrays["waveform_whole"][0], line.waveformvalues_wholedt)
    np.testing.assert_array_equal(arrays["waveform_half"][0], line.waveformvalues_halfdt)
    assert arrays["resistance"].dtype == np.float64


def test_transmission_line_host_arrays_reject_duplicate_port(float64_config):
    grid = SimpleNamespace(iterations=4, dt=0.1)

    with pytest.raises(ValueError, match="same Yee electric-field edge"):
        transmission_line_host_arrays([_line(), _line()], grid)


def test_dtoh_transmission_line_outputs(float64_config):
    line = _line()
    grid = SimpleNamespace(iterations=4, transmissionlines=[line])
    voltage = np.arange(5, dtype=np.float64).reshape(1, 5)
    current = -voltage

    dtoh_transmission_line_outputs(voltage, current, grid)

    np.testing.assert_array_equal(line.Vtotal, voltage[0])
    np.testing.assert_array_equal(line.Itotal, current[0])


def test_incident_calculation_preserves_histories_but_resets_update_state(float64_config):
    iterations = 80
    dt = 1e-12
    line = TransmissionLine(iterations, dt)
    line.resistance = 50.0
    line.waveformvalues_wholedt = np.ones(iterations + 1, dtype=np.float64)
    line.waveformvalues_halfdt = np.ones(iterations + 1, dtype=np.float64)

    # A repeated setup must not inherit any previous internal line state.
    line.voltage.fill(3.0)
    line.current.fill(-2.0)
    line.abcv0 = 4.0
    line.abcv1 = -5.0

    line.calculate_incident_V_I(SimpleNamespace(dt=dt))

    assert line.Vinc[0] == 0
    assert line.Iinc[0] == 0
    assert np.any(line.Vinc != 0)
    assert np.any(line.Iinc != 0)
    assert line.nl == line.antpos + 1
    np.testing.assert_array_equal(line.voltage, 0)
    np.testing.assert_array_equal(line.current, 0)
    assert line.abcv0 == 0
    assert line.abcv1 == 0


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize(
    "kernel",
    [
        knl_transmission_line.update_transmission_line_magnetic,
        knl_transmission_line.update_transmission_line_electric,
    ],
)
def test_transmission_line_template_substitutions_are_complete(backend, kernel):
    arguments = kernel[f"args_{backend}"].substitute({"REAL": "double"})
    body = kernel["func"].substitute(
        {
            "CUDA_IDX": "int i = 0;" if backend == "cuda" else "",
            "REAL": "double",
            "NY_TLINFO": 10,
            "NY_TLWAVES": 5,
            "NY_TLOUTPUTS": 5,
        }
    )

    assert "$" not in arguments
    assert "$" not in body
