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

from types import SimpleNamespace

import numpy as np

from toolboxes.Utilities.convert_png2h5 import Cursor


def test_cursor_records_material_on_its_instance(monkeypatch):
    monkeypatch.setattr("toolboxes.Utilities.convert_png2h5.plt.connect", lambda *args: None)
    materials = []
    image = np.array([[[0.1, 0.2, 0.3, 1.0]]])
    cursor = Cursor(image, materials)

    cursor(SimpleNamespace(dblclick=False, xdata=0, ydata=0))

    assert len(materials) == 1
    np.testing.assert_array_equal(materials[0], [25, 51, 76, 255])


def test_cursor_preserves_integer_rgb_values(monkeypatch):
    monkeypatch.setattr("toolboxes.Utilities.convert_png2h5.plt.connect", lambda *args: None)
    materials = []
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    cursor = Cursor(image, materials)

    cursor(SimpleNamespace(dblclick=False, xdata=0, ydata=0))

    np.testing.assert_array_equal(materials[0], [10, 20, 30])
