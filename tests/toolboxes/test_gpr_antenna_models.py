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

from pathlib import Path

import pytest

import gprMax
from toolboxes.GPRAntennaModels.GSSI import (
    antenna_like_GSSI_400,
    antenna_like_GSSI_1500,
    antenna_like_GSSI_2000,
)
from toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200


def test_gssi_1500_custom_optimisation_parameters_build_source():
    objects = antenna_like_GSSI_1500(
        0.5,
        0.5,
        0.1,
        absorber1Er=2,
        absorber1sig=0.1,
        absorber2Er=3,
        absorber2sig=0.2,
        pcbEr=4,
        pcbsig=0.01,
        hdpeEr=2.3,
        hdpesig=0,
    )

    assert any(isinstance(obj, gprMax.Waveform) for obj in objects)
    assert any(isinstance(obj, gprMax.VoltageSource) for obj in objects)


def test_gssi_2000_geometry_and_feed_match_published_model():
    position = (0.125, 0.125, 0.04)
    objects = antenna_like_GSSI_2000(*position)

    assert sum(isinstance(obj, gprMax.Material) for obj in objects) == 10
    assert sum(isinstance(obj, gprMax.Box) for obj in objects) == 26
    assert sum(isinstance(obj, gprMax.Plate) for obj in objects) == 69
    assert sum(isinstance(obj, gprMax.Edge) for obj in objects) == 2

    waveform = next(obj for obj in objects if isinstance(obj, gprMax.Waveform))
    source = next(obj for obj in objects if isinstance(obj, gprMax.VoltageSource))
    receiver = next(obj for obj in objects if isinstance(obj, gprMax.Rx))
    materials = {
        obj.kwargs["id"]: obj
        for obj in objects
        if isinstance(obj, gprMax.Material)
    }

    assert waveform.kwargs["amp"] == -1
    assert waveform.kwargs["freq"] == pytest.approx(2.12e9)
    assert source.point == pytest.approx((0.105, 0.124, 0.043))
    assert source.resistance == pytest.approx(560)
    assert receiver.point == pytest.approx((0.145, 0.124, 0.043))
    assert receiver.outputs == ["Ey"]
    assert materials["gssi2000_rxres"].kwargs["se"] == pytest.approx(0.0049998)
    assert materials["gssi2000_absorber1"].kwargs["se"] == pytest.approx(1.0869565)
    assert materials["gssi2000_absorber2"].kwargs["se"] == pytest.approx(1.2658228)

    geometry = [obj for obj in objects if isinstance(obj, (gprMax.Box, gprMax.Plate, gprMax.Edge))]
    points = [obj.kwargs[key] for obj in geometry for key in ("p1", "p2")]
    assert min(point[0] for point in points) == pytest.approx(position[0] - 0.043)
    assert max(point[0] for point in points) == pytest.approx(position[0] + 0.043)
    assert min(point[1] for point in points) == pytest.approx(position[1] - 0.045)
    assert max(point[1] for point in points) == pytest.approx(position[1] + 0.045)
    assert min(point[2] for point in points) == pytest.approx(position[2])
    assert max(point[2] for point in points) == pytest.approx(position[2] + 0.068)


def test_gssi_2000_rejects_unsupported_resolution():
    with pytest.raises(ValueError, match="1 mm"):
        antenna_like_GSSI_2000(0.5, 0.5, 0.1, resolution=0.002)


def test_gssi_400_rejects_unsupported_resolution():
    with pytest.raises(ValueError, match="2 mm"):
        antenna_like_GSSI_400(0.5, 0.5, 0.1, resolution=0.004)


def test_gssi_400_custom_pulse_uses_module_relative_existing_file():
    objects = antenna_like_GSSI_400(0.5, 0.5, 0.1)
    excitation = next(obj for obj in objects if isinstance(obj, gprMax.ExcitationFile))

    assert Path(excitation.filepath).is_file()


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (antenna_like_GSSI_400, {"excitationfreq": 1e9}),
        (antenna_like_MALA_1200, {"excitationfreq": 1e9}),
    ],
)
def test_partial_optimisation_parameters_are_rejected(factory, kwargs):
    with pytest.raises(ValueError, match="Missing"):
        factory(0.5, 0.5, 0.1, **kwargs)
