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
    subgrid.add(
        gprMax.VoltageSource(
            p1=source_position,
            polarisation="z",
            resistance=0,
            waveform_id="pulse",
            reference_impedance=50,
        )
    )
    port = gprMax.RxPort(
        p1=source_position,
        id="feed",
        spectrum_limit="nyquist",
    )
    subgrid.add(port)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )
    return output.with_suffix(".h5"), port, source_position


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
