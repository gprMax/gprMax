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

"""Cross-check public API hash labels, command registries, and parser dispatch."""

import inspect

import pytest

import gprMax
from gprMax.hash_cmds_file import check_cmd_names, get_user_objects
from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import MaterialList, TransmissionLine
from gprMax.user_objects.cmds_singleuse import OMPThreads
from gprMax.user_objects.user_objects import UserObject

API_ONLY_OBJECTS = {
    "PMLProps",  # Deprecated compatibility wrapper; use its two replacement commands.
    "SubGridHSG",  # Subgrids are currently defined through the Python API only.
}


def public_user_object_hashes():
    objects = []
    for name, cls in inspect.getmembers(gprMax, inspect.isclass):
        if cls is UserObject or not issubclass(cls, UserObject) or name in API_ONLY_OBJECTS:
            continue
        objects.append((name, cls.hash.fget(None)))
    return objects


@pytest.mark.parametrize("class_name, command", public_user_object_hashes())
def test_public_api_hash_label_is_accepted(class_name, command):
    try:
        check_cmd_names([f"{command}:\n"], checkessential=False)
    except SyntaxError:
        pytest.fail(f"{class_name}.hash returns unregistered command {command}")


class TrackingCommands(dict):
    def __init__(self, commands):
        super().__init__(commands)
        self.accessed = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)


def test_every_registered_multiuse_command_has_parser_dispatch():
    _, commands, _ = check_cmd_names([], checkessential=False)
    tracked = TrackingCommands(commands)

    process_multicmds(tracked)

    # Include files are expanded before multi-use user objects are constructed.
    assert tracked.accessed == set(commands) - {"#include_file"}


def test_omp_threads_hash_round_trip_uses_documented_command():
    objects = get_user_objects(["#omp_threads: 2\n"], checkessential=False)

    assert len(objects) == 1
    assert isinstance(objects[0], OMPThreads)
    assert str(objects[0]) == "#omp_threads: 2"


def test_legacy_num_threads_hash_alias_builds_canonical_api_object():
    objects = get_user_objects(["#num_threads: 2\n"], checkessential=False)

    assert len(objects) == 1
    assert isinstance(objects[0], OMPThreads)
    assert objects[0].omp_threads == 2
    assert str(objects[0]) == "#omp_threads: 2"


def test_omp_threads_and_legacy_alias_cannot_both_be_specified():
    with pytest.raises(ValueError, match="cannot both be specified"):
        get_user_objects(
            ["#omp_threads: 2\n", "#num_threads: 2\n"],
            checkessential=False,
        )


def test_material_list_hash_round_trip_does_not_become_material_range():
    objects = get_user_objects(["#material_list: mat_a mat_b mix\n"], checkessential=False)

    assert len(objects) == 1
    assert isinstance(objects[0], MaterialList)
    assert str(objects[0]) == "#material_list: mat_a mat_b mix"


def test_transmission_line_hash_optional_times_are_numeric():
    objects = get_user_objects(
        ["#transmission_line: z 0.01 0.02 0.03 50 pulse 1e-10 2e-10\n"],
        checkessential=False,
    )

    assert len(objects) == 1
    assert isinstance(objects[0], TransmissionLine)
    assert objects[0].start == pytest.approx(1e-10)
    assert objects[0].stop == pytest.approx(2e-10)


def test_time_step_stability_factor_rejects_extra_parameters():
    with pytest.raises(ValueError, match="requires exactly one parameter"):
        get_user_objects(
            ["#time_step_stability_factor: 0.9 0.8\n"],
            checkessential=False,
        )
