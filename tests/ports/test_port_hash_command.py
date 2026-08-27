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

"""Hash-command coverage for automatic voltage-source ports."""

import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.user_objects.cmds_multiuse import VoltageSource


def _parse(command):
    return get_user_objects([f"{command}\n"], checkessential=False)


@pytest.mark.parametrize(
    "command, output_id, spectrum_limit",
    [
        ("#voltage_source: z 0.1 0.2 0.3 50 pulse", None, 10),
        ("#voltage_source: z 0.1 0.2 0.3 50 pulse 0 1e-9 feed 15", "feed", 15),
        (
            "#voltage_source: z 0.1 0.2 0.3 50 pulse 0 1e-9 feed nyquist",
            "feed",
            "nyquist",
        ),
    ],
)
def test_voltage_source_port_positional_forms(command, output_id, spectrum_limit):
    objects = _parse(command)

    assert len(objects) == 1
    assert isinstance(objects[0], VoltageSource)
    assert objects[0].id == output_id
    assert objects[0].spectrum_limit == spectrum_limit


@pytest.mark.parametrize(
    "command",
    [
        "#voltage_source: z 0.1 0.2 0.3 50 pulse 0 1e-9 feed full",
        "#voltage_source: z 0.1 0.2 0.3 50 pulse 0 1e-9 feed 2",
        "#voltage_source: z 0.1 0.2 0.3 50 pulse 0 1e-9 feed nan",
    ],
)
def test_voltage_source_rejects_malformed_spectrum_limit(command):
    with pytest.raises(ValueError):
        _parse(command)


def test_voltage_source_hash_accepts_reference_impedance_after_start_stop():
    objects = _parse("#voltage_source: z 0.1 0.2 0.3 0 pulse 0 1e-9 75")

    assert len(objects) == 1
    assert isinstance(objects[0], VoltageSource)
    assert objects[0].reference_impedance == 75


def test_voltage_source_hash_keeps_reference_impedance_last_with_port_options():
    objects = _parse("#voltage_source: z 0.1 0.2 0.3 0 pulse 0 1e-9 feed nyquist 75")

    assert objects[0].id == "feed"
    assert objects[0].spectrum_limit == "nyquist"
    assert objects[0].reference_impedance == 75


def test_removed_rx_port_hash_command_is_rejected():
    with pytest.raises(SyntaxError, match="Every 3-D #voltage_source now owns its port monitor"):
        _parse("#rx_port: 0.1 0.2 0.3")
