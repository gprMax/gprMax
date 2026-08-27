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

"""Validation of Debye, Lorentz, and Drude hash-command adapters."""

from types import SimpleNamespace

import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects

VALID_COMMANDS = (
    (
        "#add_dispersion_debye: 2 1 1e-10 2 2e-10 material_a\n",
        gprMax.AddDebyeDispersion,
    ),
    (
        "#add_dispersion_lorentz: 2 1 1e9 1e8 2 2e9 2e8 material_a\n",
        gprMax.AddLorentzDispersion,
    ),
    (
        "#add_dispersion_drude: 2 1e9 1e8 2e9 2e8 material_a\n",
        gprMax.AddDrudeDispersion,
    ),
)


@pytest.mark.parametrize("command, expected_type", VALID_COMMANDS)
def test_multiple_poles_are_parsed_with_a_material_target(command, expected_type):
    objects = get_user_objects([command], checkessential=False)

    assert len(objects) == 1
    assert isinstance(objects[0], expected_type)
    assert objects[0].kwargs["poles"] == 2
    assert objects[0].kwargs["material_ids"] == ["material_a"]


@pytest.mark.parametrize(
    "command",
    (
        "#add_dispersion_debye: 2 1 1e-10 material_a\n",
        "#add_dispersion_lorentz: 2 1 1e9 1e8 material_a\n",
        "#add_dispersion_drude: 2 1e9 1e8 material_a\n",
        "#add_dispersion_debye: 2 1 1e-10 2 2e-10\n",
        "#add_dispersion_lorentz: 2 1 1e9 1e8 2 2e9 2e8\n",
        "#add_dispersion_drude: 2 1e9 1e8 2e9 2e8\n",
        "#add_dispersion_debye: 0 material_a\n",
        "#add_dispersion_lorentz: 0 material_a\n",
        "#add_dispersion_drude: 0 material_a\n",
    ),
)
def test_malformed_dispersion_commands_are_rejected_during_parsing(command):
    with pytest.raises(ValueError):
        get_user_objects([command], checkessential=False)


@pytest.mark.parametrize(
    "dispersion",
    (
        gprMax.AddDebyeDispersion(
            poles=2,
            er_delta=[1],
            tau=[1e-10, 2e-10],
            material_ids=["material_a"],
        ),
        gprMax.AddLorentzDispersion(
            poles=2,
            er_delta=[1, 2],
            omega=[1e9],
            delta=[1e8, 2e8],
            material_ids=["material_a"],
        ),
        gprMax.AddDrudeDispersion(
            poles=2,
            omega=[1e9, 2e9],
            alpha=[1e8],
            material_ids=["material_a"],
        ),
    ),
)
def test_python_api_requires_one_parameter_set_per_pole(dispersion):
    with pytest.raises(ValueError, match="one for each pole"):
        dispersion.build(SimpleNamespace(materials=[]))


@pytest.mark.parametrize(
    "dispersion",
    (
        gprMax.AddDebyeDispersion(poles=1, er_delta=[1], tau=[1e-10], material_ids=[]),
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=[1],
            omega=[1e9],
            delta=[1e8],
            material_ids=[],
        ),
        gprMax.AddDrudeDispersion(poles=1, omega=[1e9], alpha=[1e8], material_ids=[]),
    ),
)
def test_python_api_requires_a_material_target(dispersion):
    with pytest.raises(ValueError, match="at least one material identifier"):
        dispersion.build(SimpleNamespace(materials=[]))


@pytest.mark.parametrize(
    "dispersion",
    (
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=[1],
            tau=[1e-10],
            material_ids=["present", "missing"],
        ),
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=[1],
            omega=[1e9],
            delta=[1e8],
            material_ids=["present", "missing"],
        ),
        gprMax.AddDrudeDispersion(
            poles=1,
            omega=[1e9],
            alpha=[1e8],
            material_ids=["present", "missing"],
        ),
    ),
)
def test_missing_material_error_names_only_missing_identifier(dispersion):
    grid = SimpleNamespace(materials=[SimpleNamespace(ID="present")])

    with pytest.raises(ValueError, match=r"material\(s\) \['missing'\] do not exist"):
        dispersion.build(grid)
