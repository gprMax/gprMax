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

"""Small CPU integration tests for automatic voltage-source port output."""

import h5py
import numpy as np
import pytest

import gprMax


def _scene(
    spectrum_limit=10,
    background=None,
    dispersive_elsewhere=False,
    polarisation="z",
    resistance=50,
    reference_impedance=None,
):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    if background is not None:
        er, conductivity = background
        scene.add(
            gprMax.Material(
                er=er,
                se=conductivity,
                mr=1,
                sm=0,
                id="background",
            )
        )
        scene.add(
            gprMax.Box(
                p1=(0, 0, 0),
                p2=(0.02, 0.02, 0.02),
                material_id="background",
            )
        )
    if dispersive_elsewhere:
        scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="remote_debye"))
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=1,
                er_delta=(3,),
                tau=(1e-11,),
                material_ids=("remote_debye",),
            )
        )
        scene.add(
            gprMax.Box(
                p1=(0.014, 0.004, 0.004),
                p2=(0.018, 0.016, 0.016),
                material_id="remote_debye",
            )
        )
    source = gprMax.VoltageSource(
        p1=(0.01, 0.01, 0.01),
        polarisation=polarisation,
        resistance=resistance,
        waveform_id="pulse",
        id="feed",
        spectrum_limit=spectrum_limit,
        reference_impedance=reference_impedance,
    )
    scene.add(source)
    return scene, source


def _run(
    tmp_path,
    name,
    spectrum_limit=10,
    background=None,
    cpu_precision="single",
    dispersive_elsewhere=False,
    polarisation="z",
    resistance=50,
    reference_impedance=None,
):
    output = tmp_path / name
    scene, port = _scene(
        spectrum_limit,
        background,
        dispersive_elsewhere,
        polarisation,
        resistance,
        reference_impedance,
    )
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision=cpu_precision,
    )
    return str(output) + ".h5", port


def test_default_port_writes_corrected_s11_impedance_and_frequency_axis(tmp_path):
    filename, api_port = _run(tmp_path, "default_port")

    assert api_port.result.s11.size > 0

    with h5py.File(filename, "r") as output:
        assert output.attrs["nrx"] == 0
        assert output.attrs["nports"] == 1
        assert "rxs" not in output
        port = output["ports/feed"]
        frequency = port["frequency"][...]
        s11 = port["S11"][...]
        zin = port["Zin"][...]
        source_valid = port["source_valid"][...].astype(bool)
        valid_s11 = port["valid_S11"][...].astype(bool)
        valid_zin = port["valid_Zin"][...].astype(bool)

        assert frequency.dtype == np.float32
        assert s11.dtype == np.complex64
        assert port.attrs["SpectrumLimitMode"] == "minimum_wavelength_cells"
        assert port.attrs["MinimumWavelengthCells"] == 10
        assert frequency[-1] <= port.attrs["MeshFrequencyLimit"]
        np.testing.assert_allclose(port.attrs["FrequencyRange"], (frequency[0], frequency[-1]))
        assert port.attrs["BackgroundMaterial"] == "free_space"
        assert port.attrs["BackgroundConductivity"] == 0
        assert source_valid.any()
        assert valid_s11.any()
        assert valid_zin.any()
        assert port.attrs["GapCapacitance"] == pytest.approx(8.8541878128e-12 * 0.002, rel=2e-6)
        np.testing.assert_allclose(
            zin[valid_zin],
            50 * (1 + s11[valid_zin]) / (1 - s11[valid_zin]),
            rtol=2e-5,
            atol=2e-5,
        )


def test_nyquist_research_mode_retains_full_native_axis_and_validity_masks(tmp_path):
    filename, _ = _run(tmp_path, "full_port", "nyquist")

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]
        frequency = port["frequency"][...]
        iterations = int(output.attrs["Iterations"])
        expected_bins = (iterations - 1) // 2 + 1

        assert port.attrs["SpectrumLimitMode"] == "nyquist"
        assert frequency.size == expected_bins
        assert frequency[-1] > port.attrs["MeshFrequencyLimit"]
        source_valid = port["source_valid"][...].astype(bool)
        mesh_valid = port["mesh_valid"][...].astype(bool)
        gap_valid = port["gap_correction_valid"][...].astype(bool)
        assert not mesh_valid[-1]
        resolved = source_valid & mesh_valid
        assert resolved.any()
        assert gap_valid[resolved].all()
        if (iterations - 1) % 2 == 0:
            assert not gap_valid[-1]
            assert np.isnan(port["S11"][...][-1])
        np.testing.assert_allclose(port.attrs["FrequencyRange"], (frequency[0], frequency[-1]))
        for dataset in (
            "S11",
            "Zin",
            "Yin",
            "valid_S11",
            "source_valid",
            "mesh_valid",
            "cells_per_minimum_wavelength",
        ):
            assert port[dataset].shape == frequency.shape


def test_gap_properties_come_from_lossy_background_before_source_material(tmp_path):
    filename, _ = _run(tmp_path, "lossy_port", background=(4, 0.02))

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]

        assert port.attrs["BackgroundMaterial"] == "background"
        assert port.attrs["BackgroundRelativePermittivity"] == 4
        assert port.attrs["BackgroundConductivity"] == pytest.approx(0.02)
        assert port.attrs["GapCapacitance"] == pytest.approx(8.8541878128e-12 * 4 * 0.002, rel=2e-6)
        assert port.attrs["BackgroundConductance"] == pytest.approx(0.02 * 0.002)


def test_port_arrays_follow_configured_double_precision(tmp_path):
    filename, _ = _run(tmp_path, "double_port", cpu_precision="double")

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]

        assert port["frequency"].dtype == np.float64
        assert port["S11"].dtype == np.complex128
        assert port.attrs["real_dtype"] == "float64"
        assert port.attrs["complex_dtype"] == "complex128"


def test_dispersive_material_away_from_source_is_supported_and_limits_mesh(tmp_path):
    filename, _ = _run(tmp_path, "remote_debye", dispersive_elsewhere=True)

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]

        assert port.attrs["BackgroundMaterial"] == "free_space"
        assert port.attrs["LimitingMaterial"] == "remote_debye"
        assert port["valid_S11"][...].astype(bool).any()


@pytest.mark.parametrize("polarisation", ("x", "y"))
def test_other_voltage_source_polarisations_produce_valid_ports(tmp_path, polarisation):
    filename, _ = _run(
        tmp_path,
        f"{polarisation}_port",
        polarisation=polarisation,
    )

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]

        assert port.attrs["Polarisation"] == polarisation
        assert port["valid_S11"][...].astype(bool).any()


@pytest.mark.parametrize("polarisation", ("x", "y", "z"))
def test_hard_source_port_uses_phase_aligned_ampere_current(tmp_path, polarisation):
    filename, api_port = _run(
        tmp_path,
        f"hard_{polarisation}_port",
        polarisation=polarisation,
        resistance=0,
        reference_impedance=50,
    )

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]
        frequency = port["frequency"][...]
        valid = port["valid_Zin"][...].astype(bool)

        assert port.attrs["PortMode"] == "hard_delta_gap"
        assert port.attrs["ReferenceImpedance"] == 50
        assert port.attrs["ReferenceImpedanceSource"] == "voltage_source"
        assert port.attrs["TimeSampleOffset"] == pytest.approx(output.attrs["dt"])
        assert port.attrs["CurrentTimeSampleOffset"] == pytest.approx(0.5 * output.attrs["dt"])
        assert port.attrs["CurrentTimeAlignment"] == "explicit_fft_half_step_phase"
        assert port["Iloop"].shape == port["time"].shape
        assert port["time_current"].shape == port["time"].shape
        assert port["Iloop_spectrum"].shape == frequency.shape
        assert port["Iterminal_spectrum"].shape == frequency.shape
        assert valid.any()
        np.testing.assert_allclose(
            port["Zin"][...][valid],
            port["Vtotal_spectrum"][...][valid] / port["Iterminal_spectrum"][...][valid],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            api_port.result.total_voltage,
            api_port.result.generator_voltage,
        )


def test_hard_source_port_defaults_to_50_ohm_reference(tmp_path):
    filename, _ = _run(
        tmp_path,
        "hard_default_reference",
        resistance=0,
    )

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]
        assert port.attrs["ReferenceImpedance"] == 50
        assert port.attrs["ReferenceImpedanceSource"] == "voltage_source"


def test_hard_source_nyquist_gap_admittance_remains_finite(tmp_path):
    filename, _ = _run(
        tmp_path,
        "hard_nyquist_port",
        spectrum_limit="nyquist",
        resistance=0,
        reference_impedance=50,
    )

    with h5py.File(filename, "r") as output:
        port = output["ports/feed"]

        assert port["gap_correction_valid"][...].astype(bool)[-1]
        assert np.isfinite(port["gap_correction_c"][...][-1])
        assert np.isfinite(port["Iterminal_spectrum"][...][-1])
