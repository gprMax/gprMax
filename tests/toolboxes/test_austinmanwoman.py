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

"""AustinMan/AustinWoman material mapping regression tests."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from toolboxes.MaterialDatabase.convert_geometry import convert_geometry, parse_legacy_materials

TOOLBOX = Path(__file__).parents[2] / "toolboxes" / "AustinManWoman"


def _materials(filename):
    return dict(parse_legacy_materials(TOOLBOX / filename))


def _by_original_id(materials):
    return {entry["metadata"]["original_id"]: entry for entry in materials.values()}


def test_austin_material_profiles_have_identical_complete_mappings():
    constant = _materials("AustinManWoman_materials.txt")
    dispersive = _materials("AustinManWoman_materials_dispersive.txt")

    assert len(constant) == 56
    assert tuple(constant) == tuple(dispersive)
    assert [entry["metadata"]["original_id"] for entry in constant.values()] == [
        entry["metadata"]["original_id"] for entry in dispersive.values()
    ]


def test_austin_blood_debye_fit_retains_microwave_relaxation_time():
    materials = _by_original_id(_materials("AustinManWoman_materials_dispersive.txt"))
    blood = materials["Blood"]

    assert blood["model"] == "debye"
    assert blood["poles"][2]["relaxation_time_s"] == pytest.approx(0.89892e-11)

    frequency = 900e6
    omega = 2 * np.pi * frequency
    relative_permittivity = complex(blood["base"]["relative_permittivity"])
    for pole in blood["poles"]:
        relative_permittivity += pole["relative_permittivity_difference"] / (
            1 + 1j * omega * pole["relaxation_time_s"]
        )

    # The legacy non-dispersive Austin table gives er=61.36 at 900 MHz.
    assert relative_permittivity.real == pytest.approx(61.17, abs=0.02)


def test_austin_profile_converts_to_keyed_hdf5_and_json(tmp_path):
    geometry = tmp_path / "AustinMan_sample.h5"
    with h5py.File(geometry, "w") as output:
        # Austin v2.3 geometry files use this historical attribute name.
        output.attrs["dx, dy, dz"] = (0.002, 0.002, 0.002)
        output.create_dataset("data", data=np.asarray([[[0, 55]]], dtype=np.uint16))

    converted, database = convert_geometry(
        geometry,
        TOOLBOX / "AustinManWoman_materials_dispersive.txt",
    )

    with h5py.File(converted, "r") as output:
        keys = [value.decode() for value in output["material_keys"][:]]
        assert keys[0] == "material_000_Air"
        assert keys[-1] == "material_055_VitreousHumor"
        assert output.attrs["MaterialDatabase"] == database.stem
        np.testing.assert_allclose(output.attrs["dx_dy_dz"], [0.002, 0.002, 0.002])

    document = json.loads(database.read_text(encoding="utf-8"))
    assert len(document["materials"]) == 56
    assert document["materials"]["material_003_Blood"]["model"] == "debye"


pytestmark = pytest.mark.unit
