# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

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
