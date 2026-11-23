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

"""Coordinates in eigenmode port and internal PML slab log messages are plain numbers."""

import logging

import numpy as np
import pytest

import gprMax
from gprMax.user_objects import cmds_multiuse
from gprMax.user_objects.cmds_multiuse import _format_point


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def module_messages():
    """Collect this module's records.

    ``gprMax.run`` turns off propagation on the top-level ``gprMax`` logger, so
    pytest's ``caplog`` does not see them; a handler on the module logger does.
    """
    handler = _ListHandler()
    module_logger = logging.getLogger(cmds_multiuse.__name__)
    module_logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        module_logger.removeHandler(handler)


@pytest.mark.unit
def test_format_point_prints_numpy_scalars_as_numbers():
    assert _format_point((np.float64(0.005), np.float64(0.0))) == "(0.005, 0)"
    assert _format_point(np.array([0.0, 0.08, 0.06])) == "(0, 0.08, 0.06)"
    assert _format_point((np.int64(2), 1.5e-9)) == "(2, 1.5e-09)"


@pytest.mark.integration
def test_eigenmode_source_and_receiver_messages_print_plain_coordinates(tmp_path, module_messages):
    inputfile = tmp_path / "eigenmode.in"
    inputfile.write_text(
        "#domain_mode: TM\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.06 0.05 inf\n"
        "#pml_cells: 5 0 0 5 0 0\n"
        "#time_window: 4e-10\n"
        "#waveform: contsine 1 5e9 wave\n"
        "#box: 0 0 0 0.06 0.005 inf pec\n"
        "#box: 0 0.045 0 0.06 0.05 inf pec\n"
        "#eigenmode_band: band 5e9 5e9 1\n"
        "#eigenmode_port: 1 0.010 0.005 0 0.010 0.045 inf + 1 5e9 n\n"
        "#eigenmode_port: 2 0.050 0.005 0 0.050 0.045 inf - 1 5e9 n\n"
        "#eigenmode_excitation: 1 1 wave n\n"
    )

    gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "eigenmode",
        geometry_only=True,
        hide_progress_bars=True,
        log_level=30,
    )

    bounds = [message for message in module_messages if "transverse bounds" in message]
    assert any(message.startswith("Eigenmode source") for message in bounds)
    assert any(message.startswith("Eigenmode receiver") for message in bounds)
    for message in bounds:
        assert "np." not in message
        assert "transverse bounds (0.005, " in message


@pytest.mark.integration
def test_internal_pml_slab_message_prints_plain_coordinates(tmp_path, module_messages):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.PMLSlab(p1=(0, 0, 0), p2=(0.01, 0.03, 0.03), maximum_face="x0"))

    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "slab",
        hide_progress_bars=True,
        log_level=30,
    )

    (message,) = [message for message in module_messages if message.startswith("Internal PML slab")]
    assert "from (0, 0, 0) m to (0.01, 0.03, 0.03) m" in message
