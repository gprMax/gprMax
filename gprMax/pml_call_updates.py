# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
#
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

from gprMax.pml_1order_update import *
from gprMax.pml_2order_update import *


def update_pml_electric(G):
    """This functions updates electric field components with the PML correction."""

    for pml in G.pmls:
        if pml.direction == 'xminus':
            if len(pml.CFS) == 1:
                update_pml_1order_ey_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_1order_ez_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif len(pml.CFS) == 2:
                update_pml_2order_ey_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_2order_ez_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
        elif pml.direction == 'xplus':
            if len(pml.CFS) == 1:
                update_pml_1order_ey_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_1order_ez_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif len(pml.CFS) == 2:
                update_pml_2order_ey_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_2order_ez_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
        elif pml.direction == 'yminus':
            if len(pml.CFS) == 1:
                update_pml_1order_ex_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_1order_ez_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif len(pml.CFS) == 2:
                update_pml_2order_ex_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_2order_ez_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
        elif pml.direction == 'yplus':
            if len(pml.CFS) == 1:
                update_pml_1order_ex_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_1order_ez_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif len(pml.CFS) == 2:
                update_pml_2order_ex_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_2order_ez_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
        elif pml.direction == 'zminus':
            if len(pml.CFS) == 1:
                update_pml_1order_ex_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_1order_ey_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
            elif len(pml.CFS) == 2:
                update_pml_2order_ex_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_2order_ey_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
        elif pml.direction == 'zplus':
            if len(pml.CFS) == 1:
                update_pml_1order_ex_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_1order_ey_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
            elif len(pml.CFS) == 2:
                update_pml_2order_ex_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_2order_ey_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)


def update_pml_magnetic(G):
    """This functions updates magnetic field components with the PML correction."""

    for pml in G.pmls:
        if pml.direction == 'xminus':
            if len(pml.CFS) == 1:
                update_pml_1order_hy_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_1order_hz_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif len(pml.CFS) == 2:
                update_pml_2order_hy_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_2order_hz_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
        elif pml.direction == 'xplus':
            if len(pml.CFS) == 1:
                update_pml_1order_hy_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_1order_hz_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif len(pml.CFS) == 2:
                update_pml_2order_hy_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_2order_hz_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
        elif pml.direction == 'yminus':
            if len(pml.CFS) == 1:
                update_pml_1order_hx_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_1order_hz_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif len(pml.CFS) == 2:
                update_pml_2order_hx_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_2order_hz_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
        elif pml.direction == 'yplus':
            if len(pml.CFS) == 1:
                update_pml_1order_hx_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_1order_hz_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif len(pml.CFS) == 2:
                update_pml_2order_hx_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_2order_hz_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
        elif pml.direction == 'zminus':
            if len(pml.CFS) == 1:
                update_pml_1order_hx_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_1order_hy_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
            elif len(pml.CFS) == 2:
                update_pml_2order_hx_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_2order_hy_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
        elif pml.direction == 'zplus':
            if len(pml.CFS) == 1:
                update_pml_1order_hx_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_1order_hy_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
            elif len(pml.CFS) == 2:
                update_pml_2order_hx_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_2order_hy_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)


