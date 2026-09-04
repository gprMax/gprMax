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

"""End-to-end coverage for voltage-source ports inside a subgrid."""

import h5py
import numpy as np
import pytest

import gprMax


@pytest.fixture(scope="module")
def subgrid_port_result(tmp_path_factory):
    """Run one hard-source port on a 1 mm grid nested in a 3 mm grid."""

    output = tmp_path_factory.mktemp("subgrid_port") / "hard_source"
    source_position = (0.045, 0.045, 0.045)

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))

    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    source = gprMax.VoltageSource(
        p1=source_position,
        polarisation="z",
        resistance=0,
        waveform_id="pulse",
        id="feed",
        spectrum_limit="nyquist",
        reference_impedance=50,
    )
    subgrid.add(source)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )
    return output.with_suffix(".h5"), source, source_position


def test_subgrid_port_uses_owning_grid_and_returns_api_result(subgrid_port_result):
    filename, api_port, _ = subgrid_port_result

    assert api_port.result.frequency.size > 0
    assert api_port.result.valid_zin.any()

    with h5py.File(filename, "r") as output:
        assert output.attrs["nports"] == 0
        assert output.attrs["nrx"] == 0
        assert "ports" not in output
        assert "rxs" not in output

        subgrid = output["subgrids/fine_grid"]
        assert subgrid.attrs["nports"] == 1
        assert subgrid.attrs["nrx"] == 0
        assert subgrid.attrs["Iterations"] == 3 * output.attrs["Iterations"]
        np.testing.assert_allclose(subgrid.attrs["dx_dy_dz"], (0.001, 0.001, 0.001))

        port = subgrid["ports/feed"]
        assert port.attrs["PortMode"] == "hard_delta_gap"
        assert port.attrs["CellLength"] == pytest.approx(0.001)
        assert port.attrs["NyquistFrequency"] == pytest.approx(1 / (2 * subgrid.attrs["dt"]))
        assert port["time"].size == subgrid.attrs["Iterations"] - 1
        assert port["Iloop"].shape == port["time"].shape
        assert port["valid_Zin"][...].astype(bool).any()


def test_subgrid_port_and_source_store_same_global_position(subgrid_port_result):
    filename, _, source_position = subgrid_port_result

    with h5py.File(filename, "r") as output:
        subgrid = output["subgrids/fine_grid"]
        source = subgrid["srcs/src1"]
        port = subgrid["ports/feed"]

        np.testing.assert_allclose(source.attrs["Position"], source_position)
        np.testing.assert_allclose(port.attrs["Position"], source_position)
        np.testing.assert_allclose(port.attrs["Position"], source.attrs["Position"])
        np.testing.assert_array_equal(port.attrs["GridPosition"], (33, 33, 33))


def _ratio_one_hard_source_scene(*, use_subgrid):
    """Build matching uniform/HSG models for phase-sensitive port parity."""

    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=300))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.OMPThreads(1))

    owner = scene
    if use_subgrid:
        owner = gprMax.SubGridHSG(
            p1=(0.020, 0.020, 0.020),
            p2=(0.044, 0.044, 0.044),
            ratio=1,
            id="fine_grid",
        )
        scene.add(owner)

    owner.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"))
    owner.add(
        gprMax.VoltageSource(
            p1=(0.032, 0.032, 0.032),
            polarisation="z",
            resistance=0,
            waveform_id="pulse",
            id="feed",
            spectrum_limit="nyquist",
            reference_impedance=50,
        )
    )
    return scene


@pytest.mark.integration
def test_ratio_one_subgrid_hard_source_current_has_uniform_grid_phase(tmp_path):
    """The source's H-derived loop current must need no extra sample shift."""

    plain_path = tmp_path / "plain_hard_source"
    subgrid_path = tmp_path / "subgrid_hard_source"
    gprMax.run(
        scenes=[_ratio_one_hard_source_scene(use_subgrid=False)],
        outputfile=plain_path,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    gprMax.run(
        scenes=[_ratio_one_hard_source_scene(use_subgrid=True)],
        outputfile=subgrid_path,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    with h5py.File(plain_path.with_suffix(".h5"), "r") as plain, h5py.File(
        subgrid_path.with_suffix(".h5"), "r"
    ) as localised:
        reference = plain["ports/feed"]
        actual = localised["subgrids/fine_grid/ports/feed"]
        assert actual.attrs["CurrentTimeAlignment"] == "explicit_fft_half_step_phase"
        assert actual.attrs["CurrentTimeSampleOffset"] == pytest.approx(
            0.5 * localised["subgrids/fine_grid"].attrs["dt"]
        )

        for dataset in ("Vtotal", "Iloop"):
            expected = reference[dataset][:]
            observed = actual[dataset][:]
            assert np.max(np.abs(expected)) > 0
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < 1e-10

        # This spectrum is what the port current and impedance calculation
        # actually consume, so it also guards against a future sign or phase-
        # offset regression in the transform path.
        for dataset in ("Iloop_spectrum", "Iterminal_spectrum"):
            expected = reference[dataset][:]
            observed = actual[dataset][:]
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < 1e-10


def _ratio_one_transmission_line_scene(*, use_subgrid):
    """Build matching uniform/HSG transmission-line source models."""

    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=300))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.OMPThreads(1))

    owner = scene
    if use_subgrid:
        owner = gprMax.SubGridHSG(
            p1=(0.020, 0.020, 0.020),
            p2=(0.044, 0.044, 0.044),
            ratio=1,
            id="fine_grid",
        )
        scene.add(owner)

    owner.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"))
    owner.add(
        gprMax.TransmissionLine(
            p1=(0.032, 0.032, 0.032),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
        )
    )
    return scene


@pytest.mark.integration
def test_ratio_one_subgrid_transmission_line_current_has_uniform_grid_phase(tmp_path):
    """Transmission-line Itotal must retain its documented -dt/2 offset."""

    plain_path = tmp_path / "plain_line_source"
    subgrid_path = tmp_path / "subgrid_line_source"
    gprMax.run(
        scenes=[_ratio_one_transmission_line_scene(use_subgrid=False)],
        outputfile=plain_path,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    gprMax.run(
        scenes=[_ratio_one_transmission_line_scene(use_subgrid=True)],
        outputfile=subgrid_path,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        hide_progress_bars=True,
    )

    with h5py.File(plain_path.with_suffix(".h5"), "r") as plain, h5py.File(
        subgrid_path.with_suffix(".h5"), "r"
    ) as localised:
        reference = plain["tls/tl1"]
        actual = localised["subgrids/fine_grid/tls/tl1"]
        assert actual.attrs["TimeCurrentOffset"] == pytest.approx(
            -0.5 * localised["subgrids/fine_grid"].attrs["dt"]
        )

        for dataset in ("Vtotal", "Itotal", "Itotal_spectrum"):
            expected = reference[dataset][:]
            observed = actual[dataset][:]
            assert np.max(np.abs(expected)) > 0
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < 1e-10

        valid = (
            reference["valid_S11_current"][:].astype(bool)
            & actual["valid_S11_current"][:].astype(bool)
        )
        assert valid.any()
        np.testing.assert_allclose(
            actual["S11_current"][:][valid],
            reference["S11_current"][:][valid],
            rtol=1e-10,
            atol=1e-12,
        )
