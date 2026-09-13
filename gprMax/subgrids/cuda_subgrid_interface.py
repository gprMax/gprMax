# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom

# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""CUDA host layer for the HSG subgrid IS/OS interface.

Device-side counterpart of the four interface methods in
subgrids/subgrid_hsg.py:

    update_magnetic_is   6 calls to update_is
    update_electric_is   6 calls to update_is
    update_electric_os   6 calls to update_electric_os
    update_magnetic_os   6 calls to update_magnetic_os

All 24 call sites have identical shape and differ only in their constants, so
they are held here as tables rather than written out as 24 launches. The
tables are transcribed one-for-one from the Cython call sites; comparing them
against subgrid_hsg.py is the only review needed to trust this file.

Field and precursor arrays are referred to by name and resolved through
dictionaries of device pointers, so this module never owns memory - it only
launches.
"""

import numpy as np

# IS TABLES
#
# Columns match the Cython update_is() call sites:
#   (field, inc_l, inc_u, lookup, nwl, nwm, face, sign_l, sign_u, co)
#
# nwl/nwm are given as callables of (nwx, nwy, nwz) because several are
# nw* + 1. `offset` is -1 for magnetic and 0 for electric - H nodes sit one
# cell inside the boundary - and is applied per method, not per row.

IS_MAGNETIC = [
    # bottom and top
    ("Hy", "ex_bottom", "ex_top", "Hy",
     lambda x, y, z: x, lambda x, y, z: y + 1, 1, 1, -1, 3),
    ("Hx", "ey_bottom", "ey_top", "Hx",
     lambda x, y, z: x + 1, lambda x, y, z: y, 1, -1, 1, 3),
    # left and right
    ("Hz", "ey_left", "ey_right", "Hz",
     lambda x, y, z: y, lambda x, y, z: z + 1, 2, 1, -1, 1),
    ("Hy", "ez_left", "ez_right", "Hy",
     lambda x, y, z: y + 1, lambda x, y, z: z, 2, -1, 1, 1),
    # front and back
    ("Hz", "ex_front", "ex_back", "Hz",
     lambda x, y, z: x, lambda x, y, z: z + 1, 3, -1, 1, 2),
    ("Hx", "ez_front", "ez_back", "Hx",
     lambda x, y, z: x + 1, lambda x, y, z: z, 3, 1, -1, 2),
]

IS_ELECTRIC = [
    ("Ex", "hy_bottom", "hy_top", "Ex",
     lambda x, y, z: x, lambda x, y, z: y + 1, 1, 1, -1, 3),
    ("Ey", "hx_bottom", "hx_top", "Ey",
     lambda x, y, z: x + 1, lambda x, y, z: y, 1, -1, 1, 3),
    ("Ey", "hz_left", "hz_right", "Ey",
     lambda x, y, z: y, lambda x, y, z: z + 1, 2, 1, -1, 1),
    ("Ez", "hy_left", "hy_right", "Ez",
     lambda x, y, z: y + 1, lambda x, y, z: z, 2, -1, 1, 1),
    ("Ex", "hz_front", "hz_back", "Ex",
     lambda x, y, z: x, lambda x, y, z: z + 1, 3, -1, 1, 2),
    ("Ez", "hx_front", "hx_back", "Ez",
     lambda x, y, z: x + 1, lambda x, y, z: z, 3, 1, -1, 2),
]


# OS TABLES
#
# Columns match the Cython update_*_os() call sites:
#   (face, bounds, nwn_axis, main_field, sub_field, co, sign_n, sign_f, mid)
#
# `bounds` names the six loop limits symbolically; they are resolved against
# the i/j/k lower and upper OS positions, with "+1" where the Cython adds one
# and "-1" where the magnetic table subtracts one from the near normal index.

OS_ELECTRIC = [
    (3, ("i_l", "i_u", "k_l", "k_u+1", "j_l", "j_u"), "y", "Ex", "Hz", 2, 1, -1, 1),
    (3, ("i_l", "i_u+1", "k_l", "k_u", "j_l", "j_u"), "y", "Ez", "Hx", 2, -1, 1, 0),
    (2, ("j_l", "j_u", "k_l", "k_u+1", "i_l", "i_u"), "x", "Ey", "Hz", 1, -1, 1, 1),
    (2, ("j_l", "j_u+1", "k_l", "k_u", "i_l", "i_u"), "x", "Ez", "Hy", 1, 1, -1, 0),
    (1, ("i_l", "i_u", "j_l", "j_u+1", "k_l", "k_u"), "z", "Ex", "Hy", 3, -1, 1, 1),
    (1, ("i_l", "i_u+1", "j_l", "j_u", "k_l", "k_u"), "z", "Ey", "Hx", 3, 1, -1, 0),
]

OS_MAGNETIC = [
    (3, ("i_l", "i_u", "k_l", "k_u+1", "j_l-1", "j_u"), "y", "Hz", "Ex", 2, 1, -1, 1),
    (3, ("i_l", "i_u+1", "k_l", "k_u", "j_l-1", "j_u"), "y", "Hx", "Ez", 2, -1, 1, 0),
    (2, ("j_l", "j_u", "k_l", "k_u+1", "i_l-1", "i_u"), "x", "Hz", "Ey", 1, -1, 1, 1),
    (2, ("j_l", "j_u+1", "k_l", "k_u", "i_l-1", "i_u"), "x", "Hy", "Ez", 1, 1, -1, 0),
    (1, ("i_l", "i_u", "j_l", "j_u+1", "k_l-1", "k_u"), "z", "Hy", "Ex", 3, -1, 1, 1),
    (1, ("i_l", "i_u+1", "j_l", "j_u", "k_l-1", "k_u"), "z", "Hx", "Ey", 3, 1, -1, 0),
]


class CUDASubgridInterface:
    """Launches the six IS/OS interface kernels for one subgrid.

    Args:
        kernels: dict with built CUDA functions under the keys
                    'update_is', 'update_electric_os', 'update_magnetic_os'.
        geom: SubgridGeometry describing sizes and OS placement.
        tpb: threads per block.
    """

    def __init__(self, kernels, geom, tpb=128):
        self.k_is = kernels["update_is"]
        self.k_e_os = kernels["update_electric_os"]
        self.k_m_os = kernels["update_magnetic_os"]
        self.g = geom
        self.tpb = tpb

    def _grid(self, total):
        # PyCUDA rejects numpy integers as launch dimensions. Sub-grid
        # extents come from rounded main-grid indices, so they arrive
        # as numpy.int32.
        total = int(total)
        return ((total + self.tpb - 1) // self.tpb, 1, 1)

    # ------------------------------------------------------------ IS
    def _run_is(self, table, offset, coeffs_dev, sub_fields, precursors):
        """Args:
            table: IS_MAGNETIC or IS_ELECTRIC.
            offset: -1 for magnetic, 0 for electric.
            coeffs_dev: device pointer to the subgrid's update coefficients.
            sub_fields: dict name -> device pointer, subgrid field arrays.
            precursors: dict name -> (device pointer, ny) for precursor slices.
        """
        g = self.g
        for (fname, l_name, u_name, lookup,
             f_nwl, f_nwm, face, sign_l, sign_u, co) in table:

            nwl = f_nwl(g.nwx, g.nwy, g.nwz)
            nwm = f_nwm(g.nwx, g.nwy, g.nwz)
            inc_l, inc_ny = precursors[l_name]
            inc_u, _ = precursors[u_name]
            total = nwl * nwm

            self.k_is(
                np.int32(g.nwx), np.int32(g.nwy), np.int32(g.nwz),
                np.int32(g.n_boundary_cells), np.int32(offset),
                np.int32(nwl), np.int32(nwm), np.int32(face),
                np.int32(sign_l), np.int32(sign_u), np.int32(co),
                np.int32(g.id_lookup[lookup]), np.int32(g.ny_matcoeffs),
                np.int32(g.sub_nx), np.int32(g.sub_ny), np.int32(g.sub_nz),
                np.int32(g.sub_nx), np.int32(g.sub_ny), np.int32(g.sub_nz),
                np.int32(inc_ny),
                coeffs_dev, g.sub_id, sub_fields[fname], inc_l, inc_u,
                block=(int(self.tpb), 1, 1), grid=self._grid(total),
            )

    def update_magnetic_is(self, coeffs_h_dev, sub_fields, precursors):
        self._run_is(IS_MAGNETIC, -1, coeffs_h_dev, sub_fields, precursors)

    def update_electric_is(self, coeffs_e_dev, sub_fields, precursors):
        self._run_is(IS_ELECTRIC, 0, coeffs_e_dev, sub_fields, precursors)

    # ------------------------------------------------------------ OS
    def _resolve(self, expr, bounds):
        """Turn a bound name like 'k_u+1' or 'j_l-1' into an integer."""
        if expr.endswith("+1"):
            return bounds[expr[:-2]] + 1
        if expr.endswith("-1"):
            return bounds[expr[:-2]] - 1
        return bounds[expr]

    def _run_os(self, kernel, table, coeffs_dev, main_fields, sub_fields):
        """Args:
            kernel: the built update_electric_os or update_magnetic_os.
            table: OS_ELECTRIC or OS_MAGNETIC.
            coeffs_dev: device pointer to the MAIN grid update coefficients.
            main_fields: dict name -> device pointer, main grid field arrays.
            sub_fields: dict name -> device pointer, subgrid field arrays.
        """
        g = self.g
        b = g.os_bounds()
        nwn_of = {"x": g.nwx, "y": g.nwy, "z": g.nwz}

        for (face, names, nwn_axis, main_name, sub_name,
             co, sign_n, sign_f, mid) in table:

            l_l = self._resolve(names[0], b)
            l_u = self._resolve(names[1], b)
            m_l = self._resolve(names[2], b)
            m_u = self._resolve(names[3], b)
            n_l = self._resolve(names[4], b)
            n_u = self._resolve(names[5], b)
            total = (l_u - l_l) * (m_u - m_l)

            kernel(
                np.int32(face),
                np.int32(l_l), np.int32(l_u), np.int32(m_l), np.int32(m_u),
                np.int32(n_l), np.int32(n_u),
                np.int32(nwn_of[nwn_axis]),
                np.int32(g.main_id_lookup[main_name]), np.int32(co),
                np.int32(sign_n), np.int32(sign_f), np.int32(mid),
                np.int32(g.ratio), np.int32(g.is_os_sep),
                np.int32(g.n_boundary_cells), np.int32(g.ny_matcoeffs),
                np.int32(g.main_nx), np.int32(g.main_ny), np.int32(g.main_nz),
                np.int32(g.main_ny), np.int32(g.main_nz),
                np.int32(g.sub_ny), np.int32(g.sub_nz),
                coeffs_dev, g.main_id,
                main_fields[main_name], sub_fields[sub_name],
                block=(int(self.tpb), 1, 1), grid=self._grid(total),
            )

    def update_electric_os(self, coeffs_e_dev, main_fields, sub_fields):
        self._run_os(self.k_e_os, OS_ELECTRIC, coeffs_e_dev,
                     main_fields, sub_fields)

    def update_magnetic_os(self, coeffs_h_dev, main_fields, sub_fields):
        self._run_os(self.k_m_os, OS_MAGNETIC, coeffs_h_dev,
                     main_fields, sub_fields)


class SubgridGeometry:
    """Sizes, placement and lookup tables needed to launch the interface.

    Everything here is fixed at setup, so an instance is built once per
    subgrid and reused for the whole run.
    """

    def __init__(self, *, nwx, nwy, nwz, ratio, is_os_sep, n_boundary_cells,
                 i0, j0, k0, i1, j1, k1,
                 sub_shape, main_shape, sub_id, main_id,
                 id_lookup, main_id_lookup, ny_matcoeffs):
        self.nwx, self.nwy, self.nwz = nwx, nwy, nwz
        self.ratio = ratio
        self.is_os_sep = is_os_sep
        self.n_boundary_cells = n_boundary_cells
        self.i0, self.j0, self.k0 = i0, j0, k0
        self.i1, self.j1, self.k1 = i1, j1, k1
        self.sub_nx, self.sub_ny, self.sub_nz = sub_shape
        self.main_nx, self.main_ny, self.main_nz = main_shape
        self.sub_id = sub_id
        self.main_id = main_id
        self.id_lookup = id_lookup
        self.main_id_lookup = main_id_lookup
        self.ny_matcoeffs = ny_matcoeffs

    def os_bounds(self):
        """The i/j/k Outer Surface limits, as in subgrid_hsg.py."""
        s = self.is_os_sep
        return {
            "i_l": self.i0 - s, "i_u": self.i1 + s,
            "j_l": self.j0 - s, "j_u": self.j1 + s,
            "k_l": self.k0 - s, "k_u": self.k1 + s,
        }

    def check(self):
        """Guard against the geometry that silently reads out of bounds.

        The electric OS kernel evaluates nb - s*r - r + r//2 as a subgrid
        index. A model with too few boundary cells makes that negative, and
        the kernel would read before the start of the array rather than
        failing. The Cython has the same exposure; on the device it is worth
        turning into an explicit error.
        """
        r, s, nb = self.ratio, self.is_os_sep, self.n_boundary_cells
        n_s_l = nb - s * r - r + r // 2
        if n_s_l < 0:
            raise ValueError(
                f"subgrid geometry gives a negative OS index "
                f"(nb={nb}, s={s}, r={r} -> {n_s_l}); "
                f"need n_boundary_cells >= {s * r + r - r // 2}"
            )
        return True
