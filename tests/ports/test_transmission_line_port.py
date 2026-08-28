# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Automatic S11 and impedance output for transmission-line sources."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.materials import Material
from gprMax.ports import TransmissionLinePortOutput


@pytest.fixture
def port_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            em_consts={
                "c": 299792458.0,
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
            },
        ),
    )


def test_discrete_line_current_deembedding_recovers_known_reflection(port_config):
    """The current check must correct both Yee time and line-space offsets."""

    nsamples = 256
    dt = 1e-12
    frequency_bin = 8
    omega_step = 2 * np.pi * frequency_bin / nsamples
    line_dl = np.sqrt(3) * config.sim_config.em_consts["c"] * dt
    line_k_step = 2 * np.arcsin(np.sqrt(3) * np.sin(omega_step / 2))
    reference_impedance = 50.0
    reflection = 0.25
    sample = np.arange(nsamples)

    incident_voltage = np.cos(omega_step * sample)
    reflected_voltage = reflection * np.cos(omega_step * sample)
    incident_current = np.cos(omega_step * (sample - 0.5) - line_k_step / 2) / reference_impedance
    reflected_current = (
        -reflection * np.cos(omega_step * (sample - 0.5) + line_k_step / 2) / reference_impedance
    )

    def legacy_history(values):
        return np.concatenate((values, np.zeros(1)))

    source = SimpleNamespace(
        resistance=reference_impedance,
        dl=line_dl,
        polarisation="z",
        waveformID="wave",
        Vinc=legacy_history(incident_voltage),
        Iinc=legacy_history(incident_current),
        Vtotal=legacy_history(incident_voltage + reflected_voltage),
        Itotal=legacy_history(incident_current + reflected_current),
    )
    free_space = Material(0, "free_space")
    grid = SimpleNamespace(
        iterations=nsamples,
        dt=dt,
        dx=1e-4,
        dy=1e-4,
        dz=1e-4,
        materials=[free_space],
    )

    result = TransmissionLinePortOutput(source, 1).finalise(grid)
    index = np.flatnonzero(
        np.isclose(
            result.frequency,
            frequency_bin / (nsamples * dt),
            rtol=1e-12,
        )
    )[0]

    assert result.valid_s11[index]
    assert result.valid_s11_current[index]
    assert result.s11[index] == pytest.approx(reflection, abs=1e-12)
    assert result.s11_current[index] == pytest.approx(reflection, abs=1e-12)
    expected_impedance = reference_impedance * (1 + reflection) / (1 - reflection)
    assert result.zin[index] == pytest.approx(expected_impedance, abs=1e-10)
    assert result.zin_current[index] == pytest.approx(expected_impedance, abs=1e-10)


def _scene():
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.012, 0.012, 0.012)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="w"))
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z",
            p1=(0.006, 0.006, 0.006),
            resistance=50,
            waveform_id="w",
        )
    )
    return scene


def test_cpu_line_writes_automatic_s11_and_impedance(tmp_path):
    output_path = tmp_path / "tl_port"
    gprMax.run(
        scenes=[_scene()],
        n=1,
        outputfile=output_path,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    with h5py.File(str(output_path) + ".h5", "r") as output:
        line = output["tls/tl1"]
        frequency = line["frequency"][...]
        s11 = line["S11"][...]
        zin = line["Zin"][...]
        valid = line["valid_Zin"][...].astype(bool)
        valid_current = valid & line["valid_Zin_current"][...].astype(bool)

        assert frequency.dtype == np.float32
        assert s11.dtype == np.complex64
        assert line.attrs["ZinPrimaryMethod"] == "voltage_wave_S11"
        assert line.attrs["CurrentCheckMethod"] == "discrete_line_wave_deembedding"
        assert line.attrs["MinimumWavelengthCells"] == 10
        assert valid.any()
        np.testing.assert_allclose(
            zin[valid],
            50 * (1 + s11[valid]) / (1 - s11[valid]),
            rtol=2e-5,
            atol=2e-5,
        )
        assert valid_current.any()
        assert (
            np.linalg.norm(line["S11_current"][...][valid_current] - s11[valid_current])
            / np.linalg.norm(s11[valid_current])
            < 0.05
        )
        for name in (
            "Yin",
            "S11_current",
            "Zin_current",
            "valid_S11_current",
            "valid_Zin_current",
            "source_valid",
            "mesh_valid",
            "line_propagation_valid",
        ):
            assert line[name].shape == frequency.shape
