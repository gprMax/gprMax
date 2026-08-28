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

"""Regression tests for gprMax logging utilities."""

import logging

import pytest

from gprMax.utilities.logging import CustomFormatter, logging_config


def test_custom_formatter_handles_parameterised_messages():
    record = logging.LogRecord(
        name="gprMax.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="surface %s has %d patches",
        args=("box", 42),
        exc_info=None,
    )

    formatted = CustomFormatter("%(levelname)s: %(message)s").format(record)

    assert "surface box has 42 patches" in formatted


def test_logging_config_rejects_unknown_format_style():
    with pytest.raises(ValueError, match="format_style"):
        logging_config(format_style="unknown")
