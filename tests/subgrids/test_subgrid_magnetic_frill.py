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

"""End-to-end coverage for a magnetic-frill source inside a subgrid."""

import h5py
import numpy as np
import pytest

import gprMax


@pytest.fixture(scope="module")
def subgrid_frill_result(tmp_path_factory):
    output = tmp_path_factory.mktemp("subgrid_frill") / "frill"
    feed = (0.045, 0.045, 0.045)

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
        gprMax.Plate(
            p1=(0.035, 0.035, feed[2]),
            p2=(0.055, 0.055, feed[2]),
            material_id="pec",
        )
    )
    subgrid.add(
        gprMax.ThinWire(
            p1=feed,
            p2=(feed[0], feed[1], 0.055),
            radius=0.1e-3,
        )
    )
    subgrid.add(
        gprMax.MagneticFrillSource(
            p1=feed,
            polarisation="z",
            zcoax=50,
            waveform_id="pulse",
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )
    return output.with_suffix(".h5"), feed


def test_subgrid_frill_uses_fine_grid_and_writes_automatic_port(subgrid_frill_result):
    filename, _ = subgrid_frill_result

    with h5py.File(filename, "r") as output:
        assert output.attrs["nsrc"] == 0
        assert "frills" not in output

        subgrid = output["subgrids/fine_grid"]
        assert subgrid.attrs["nsrc"] == 1
        assert subgrid.attrs["Iterations"] == 3 * output.attrs["Iterations"]
        np.testing.assert_allclose(subgrid.attrs["dx_dy_dz"], (0.001, 0.001, 0.001))

        frill = subgrid["frills/frill1"]
        assert frill.attrs["Polarisation"] == "z"
        assert frill.attrs["Z0"] == pytest.approx(50)
        assert frill.attrs["InnerConductorRadius"] == pytest.approx(0.1e-3)
        assert frill.attrs["NyquistFrequency"] == pytest.approx(1 / (2 * subgrid.attrs["dt"]))
        assert frill["time"].size == subgrid.attrs["Iterations"]
        assert frill["Vinc"].size == subgrid.attrs["Iterations"] + 1
        for name in ("Vtotal", "Itot", "S11", "Zin", "Yin"):
            assert np.all(np.isfinite(frill[name][...]))
        assert np.max(np.abs(frill["Vtotal"][...])) > 0


def test_subgrid_frill_position_is_global_and_terminal_identity_holds(
    subgrid_frill_result,
):
    filename, feed = subgrid_frill_result

    with h5py.File(filename, "r") as output:
        frill = output["subgrids/fine_grid/frills/frill1"]
        np.testing.assert_allclose(frill.attrs["Position"], feed)
        np.testing.assert_allclose(
            frill["Vtotal"][...],
            2 * frill["Vinc"][...] - frill.attrs["Z0"] * frill["Itot"][...],
            rtol=2e-5,
            atol=5e-8,
        )
