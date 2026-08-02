# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

from pathlib import Path

import pytest

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400, antenna_like_GSSI_1500
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
