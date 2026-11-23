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

import h5py

import gprMax

INPUT = """#title: dotted geometry output names
#domain: 0.1 0.1 0.1
#dx_dy_dz: 0.005 0.005 0.005
#time_window: 1e-12
#pml_cells: 0
#material: 4 0 1 0 half_space
#box: 0 0 0 0.1 0.05 0.1 half_space
#geometry_view: 0 0 0 0.1 0.05 0.1 0.005 0.005 0.005 view_0.1m n
#geometry_view: 0 0.05 0 0.1 0.1 0.1 0.005 0.005 0.005 view_0.2m n
#geometry_objects_write: 0 0 0 0.1 0.05 0.1 layer_0.1m
#geometry_objects_write: 0 0.05 0 0.1 0.1 0.1 layer_0.2m
"""


def test_dotted_geometry_output_names_do_not_overwrite_each_other(tmp_path):
    inputfile = tmp_path / "dotted.in"
    inputfile.write_text(INPUT)
    gprMax.run(inputfile=str(inputfile), n=2, geometry_only=True, hide_progress_bars=True)

    views = sorted(path.name for path in tmp_path.glob("*.vtkhdf"))
    assert views == [
        "view_0.1m1.vtkhdf",
        "view_0.1m2.vtkhdf",
        "view_0.2m1.vtkhdf",
        "view_0.2m2.vtkhdf",
    ]
    objects = sorted(path.name for path in tmp_path.glob("*.h5"))
    assert objects == ["layer_0.1m.h5", "layer_0.2m.h5"]
    for name, database in [
        ("layer_0.1m.h5", "layer_0_1m_materials"),
        ("layer_0.2m.h5", "layer_0_2m_materials"),
    ]:
        with h5py.File(tmp_path / name) as geometry:
            assert geometry.attrs["MaterialDatabase"] == database
