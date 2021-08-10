# Copyright (C) 2015-2021: The University of Edinburgh
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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


import logging
import sys

from pytools import partition

import gprMax.config as config
from ..cython.fields_updates_hsg import (cython_update_electric_os,
                                         cython_update_is,
                                         cython_update_magnetic_os)
from .base import SubGridBase

from string import Template
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

kernel_template_is = Template("""
// Defining Macros
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS) + (j)*($NZ_FIELDS) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID) + (i)*($NY_ID)*($NZ_ID) + (j)*($NZ_ID) + (k)

__global__ void hsg_update_is(
    const int nwx, const int nwy, const int nwz,
    const int n,
    const int offset,
    const int nwl, const int nwm,
    const int face,
    const int co,
    const int sign_l, const int sign_u,
    const unsigned int lookup_id,
    const int pre_coeff,
    $REAL* updatecoeffs,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field_l,
    $REAL* inc_field_u) {
        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int l, m, i1, j1, k1, i2, j2, k2, field_material_l, field_material_u, inc_i, inc_j;
        double inc_l, inc_u, f_l, f_u;
        int n_o;

        // For inner Faces H nodes are 1 cell before n_boundary_cell
        n_o = n + offset;

        // Linear Index to subscript 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        if(l >= n && l < (nwl + n) && m >= n && m < (nwm + n)) {
            if(face == 1) {
                i1 = l; j1 = m; k1 = n_o;
                i2 = l; j2 = m; k2 = n + nwz;
            }
            else if(face == 2) {
                i1 = n_o; j1 = l; k1 = m;
                i2 = n + nwx; j2 = l; k2 = m;
            }
            else {
                i1 = l; j1 = n_o; k1 = m;
                i2 = l; j2 = n + nwy; k2 = m;
            }
            
            inc_i = l - n;
            inc_j = m - n;

            // printf("(%d,%d)\\t", l, m);

            // Precursor Field index
            int pre_index = (inc_i * pre_coeff) + inc_j;

            field_material_l = ID[INDEX4D_ID(lookup_id, i1, j1, k1)];
            inc_l = inc_field_l[pre_index];

            // Additional Field at i, j, k
            f_l = updatecoeffs[INDEX2D_MAT(field_material_l, co)] * inc_l * sign_l;
            
            // Setting the new value
            field[INDEX3D_FIELDS(i1, j1, k1)] += f_l;

            field_material_u = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_u = inc_field_u[pre_index];

            f_u = updatecoeffs[INDEX2D_MAT(field_material_u, co)] * inc_u * sign_u;
            field[INDEX3D_FIELDS(i2, j2, k2)] += f_u;
        }
    }
""")


kernel_template_os = Template("""
// Defining Macros
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS) + (j)*($NZ_FIELDS) + (k)
#define INDEX3D_SUBFIELDS(i, j, k) (i)*($NY_SUBFIELDS)*($NZ_SUBFIELDS) + (j)*($NZ_SUBFIELDS) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID) + (i)*($NY_ID)*($NZ_ID) + (j)*($NZ_ID) + (k)


__global__ void hsg_update_electric_os(
    const int face,
    const unsigned int co,
    const int sign_n, const int sign_f,
    const int mid,
    const int sub_ratio,
    const int surface_sep,
    const int n_boundary_cells,
    const int nwn,
    const unsigned int lookup_id,
    const int l_l, const int l_u,
    const int m_l, const int m_u,
    const int n_l, const int n_u,
    $REAL* updatecoeffsE,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field) {
        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int l, m, l_s, m_s, n_s_l, n_s_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
        int os;
        double inc_n, inc_f;
        
        // Surface normal index for the subgrid near face h nodes (left i index)
        n_s_l = n_boundary_cells - (surface_sep * sub_ratio) - sub_ratio + floor((double) sub_ratio / 2);

        // surface normal index for the subgrid far face h nodes (right i index)
        n_s_r = n_boundary_cells + nwn + (surface_sep * sub_ratio) + floor((double) sub_ratio / 2);

        // OS at the left face
        os = n_boundary_cells - (sub_ratio * surface_sep);

        // Linear Index to subscript 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        if(face == 3 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

                // Main grid Index
            i0 = l; j0 = n_l; k0 = m;
            
            // Sub-grid Index
            i1 = l_s; j1 = n_s_l; k1 = m_s;
            i2 = l; j2 = n_u; k2 = m;
            i3 = l_s; j3 = n_s_r; k3 = m_s;

            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;

            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;

            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        if(face == 2 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = n_l; j0 = l; k0 = m;

            // Sub-grid Index
            i1 = n_s_l; j1 = l_s; k1 = m_s;
            i2 = n_u; j2 = l; k2 = m;
            i3 = n_s_r; j3 = l_s; k3 = m_s;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        if(face == 1 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = m; k0 = n_l;

            // Sub-grid Index
            i1 = l_s; j1 = m_s; k1 = n_s_l;
            i2 = l; j2 = m; k2 = n_u;
            i3 = l_s; j3 = m_s; k3 = n_s_r;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }
    }

__global__ void hsg_update_magnetic_os(
    const int face,
    const unsigned int co,
    const int sign_n, const int sign_f,
    const int mid,
    const int sub_ratio,
    const int surface_sep,
    const int n_boundary_cells,
    const int nwn,
    const unsigned int lookup_id,
    const int l_l, const int l_u,
    const int m_l, const int m_u,
    const int n_l, const int n_u,
    $REAL* updatecoeffsH,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field) {
        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        int l, m, l_s, m_s, n_s_l, n_s_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
        int os;
        double inc_n, inc_f;

        // Index (normal to os) for the subgrid near face e node
        n_s_l = n_boundary_cells - sub_ratio * surface_sep;

        // Normal index for the subgrid far face e node
        n_s_r = n_boundary_cells + nwn + sub_ratio * surface_sep;

        // OS Inner index for the subgrid
        os = n_boundary_cells - sub_ratio * surface_sep;

        // Linear Index to subscript 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        if(face == 3 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords for front face
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                // subgrid coords for back face
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = n_l; k0 = m;
            
            // Sub-grid Index
            i1 = l_s; j1 = n_s_l; k1 = m_s;
            i2 = l; j2 = n_u; k2 = m;
            i3 = l_s; j3 = n_s_r; k3 = m_s;
            
            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;               
        }

        if(face == 2 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = n_l; j0 = l; k0 = m;

            // Sub-grid Index
            i1 = n_s_l; j1 = l_s; k1 = m_s;
            i2 = n_u; j2 = l; k2 = m;
            i3 = n_s_r; j3 = l_s; k3 = m_s;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        if(face == 1 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = m; k0 = n_l;

            // Sub-grid Index
            i1 = l_s; j1 = m_s; k1 = n_s_l;
            i2 = l; j2 = m; k2 = n_u;
            i3 = l_s; j3 = m_s; k3 = n_s_r;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }
    }
""")

class SubGridHSG(SubGridBase):

    gridtype = '3DSUBGRID'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gridtype = SubGridHSG.gridtype

    def update_magnetic_is(self, precursors):
        """Update the subgrid nodes at the IS with the currents derived
            from the main grid.

        Args:
            nwl, nwm, nwn, face, field, inc_field, lookup_id, sign, mod, co
        """

        # Allocating GPU Arrays
        self.updatecoeffsH_gpu = gpuarray.to_gpu(self.updatecoeffsH)
        self.ID_gpu = gpuarray.to_gpu(self.ID)

        self.Hx_gpu = gpuarray.to_gpu(self.Hx)
        self.Hy_gpu = gpuarray.to_gpu(self.Hy)
        self.Hz_gpu = gpuarray.to_gpu(self.Hz)

        precursors.ex_bottom_gpu = gpuarray.to_gpu(precursors.ex_bottom)
        precursors.ex_top_gpu = gpuarray.to_gpu(precursors.ex_top)
        precursors.ex_front_gpu = gpuarray.to_gpu(precursors.ex_front)
        precursors.ex_back_gpu = gpuarray.to_gpu(precursors.ex_back)

        precursors.ey_bottom_gpu = gpuarray.to_gpu(precursors.ey_bottom)
        precursors.ey_top_gpu = gpuarray.to_gpu(precursors.ey_top)
        precursors.ey_left_gpu = gpuarray.to_gpu(precursors.ey_left)
        precursors.ey_right_gpu = gpuarray.to_gpu(precursors.ey_right)

        precursors.ez_front_gpu = gpuarray.to_gpu(precursors.ez_front)
        precursors.ez_back_gpu = gpuarray.to_gpu(precursors.ez_back)
        precursors.ez_left_gpu = gpuarray.to_gpu(precursors.ez_left)
        precursors.ez_right_gpu = gpuarray.to_gpu(precursors.ez_right)
        
        mod = SourceModule(kernel_template_is.substitute(
            REAL = config.sim_config.dtypes['C_float_or_double'],
            NY_MATCOEFFS = self.updatecoeffsH.shape[1],
            NX_FIELDS = self.nx + 1,
            NY_FIELDS = self.ny + 1,
            NZ_FIELDS = self.nz + 1,
            NX_ID = self.ID.shape[1],
            NY_ID = self.ID.shape[2],
            NZ_ID = self.ID.shape[3]
        ))

        hsg_update_is_gpu = mod.get_function("hsg_update_is")

        bpg = (int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)) / 128)), 1, 1)

        # Bottom and Top
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwx), np.int32(self.nwy + 1),
            np.int32(1), # face
            np.int32(3), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Hy']),
            np.int32(precursors.ex_bottom.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hy_gpu.gpudata,
            precursors.ex_bottom_gpu.gpudata,
            precursors.ex_top_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwx + 1), np.int32(self.nwy),
            np.int32(1), # face
            np.int32(3), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Hx']),
            np.int32(precursors.ey_bottom.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hx_gpu.gpudata,
            precursors.ey_bottom_gpu.gpudata,
            precursors.ey_top_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Left and Right
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwy), np.int32(self.nwz + 1),
            np.int32(2), # face
            np.int32(1), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Hz']),
            np.int32(precursors.ey_left.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hz_gpu.gpudata,
            precursors.ey_left_gpu.gpudata,
            precursors.ey_right_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwy + 1), np.int32(self.nwz),
            np.int32(2), # face
            np.int32(1), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Hy']),
            np.int32(precursors.ez_left.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hy_gpu.gpudata,
            precursors.ez_left_gpu.gpudata,
            precursors.ez_right_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Front and back
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwx), np.int32(self.nwz + 1),
            np.int32(3), # face
            np.int32(2), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Hz']),
            np.int32(precursors.ex_front.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hz_gpu.gpudata,
            precursors.ex_front_gpu.gpudata,
            precursors.ex_back_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(-1), # offset
            np.int32(self.nwx + 1), np.int32(self.nwz),
            np.int32(3), # face
            np.int32(2), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Hx']),
            np.int32(precursors.ez_front.shape[1]), # precursor node coefficient
            self.updatecoeffsH_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Hx_gpu.gpudata,
            precursors.ez_front_gpu.gpudata,
            precursors.ez_back_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        self.Hx = self.Hx_gpu.get()
        self.Hy = self.Hy_gpu.get()
        self.Hz = self.Hz_gpu.get()

        # Hz = c0Hz - c1Ey + c2Ex
        # Hy = c0Hy - c3Ex + c1Ez
        # Hx = c0Hx - c2Ez + c3Ey
        # bottom and top
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwx, self.nwy + 1, self.nwz, 1, self.Hy, precursors.ex_bottom, precursors.ex_top, self.IDlookup['Hy'], 1, -1, 3, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwx + 1, self.nwy, self.nwz, 1, self.Hx, precursors.ey_bottom, precursors.ey_top, self.IDlookup['Hx'], -1, 1, 3, config.get_model_config().ompthreads)

        # left and right
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwy, self.nwz + 1, self.nwx, 2, self.Hz, precursors.ey_left, precursors.ey_right, self.IDlookup['Hz'], 1, -1, 1, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwy + 1, self.nwz, self.nwx, 2, self.Hy, precursors.ez_left, precursors.ez_right, self.IDlookup['Hy'], -1, 1, 1, config.get_model_config().ompthreads)

        # front and back
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwx, self.nwz + 1, self.nwy, 3, self.Hz, precursors.ex_front, precursors.ex_back, self.IDlookup['Hz'], -1, 1, 2, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsH, self.ID, self.n_boundary_cells, -1, self.nwx + 1, self.nwz, self.nwy, 3, self.Hx, precursors.ez_front, precursors.ez_back, self.IDlookup['Hx'], 1, -1, 2, config.get_model_config().ompthreads)

    def update_electric_is(self, precursors):
        """Update the subgrid nodes at the IS with the currents derived
            from the main grid.

        Args:
            nwl, nwm, nwn, face, field, inc_field, lookup_id, sign, mod, co
        """

        # Allocating GPU Arrays
        self.updatecoeffsE_gpu = gpuarray.to_gpu(self.updatecoeffsE)
        self.ID_gpu = gpuarray.to_gpu(self.ID)

        self.Ex_gpu = gpuarray.to_gpu(self.Ex)
        self.Ey_gpu = gpuarray.to_gpu(self.Ey)
        self.Ez_gpu = gpuarray.to_gpu(self.Ez)

        precursors.hx_bottom_gpu = gpuarray.to_gpu(precursors.hx_bottom)
        precursors.hx_top_gpu = gpuarray.to_gpu(precursors.hx_top)
        precursors.hx_front_gpu = gpuarray.to_gpu(precursors.hx_front)
        precursors.hx_back_gpu = gpuarray.to_gpu(precursors.hx_back)

        precursors.hy_bottom_gpu = gpuarray.to_gpu(precursors.hy_bottom)
        precursors.hy_top_gpu = gpuarray.to_gpu(precursors.hy_top)
        precursors.hy_left_gpu = gpuarray.to_gpu(precursors.hy_left)
        precursors.hy_right_gpu = gpuarray.to_gpu(precursors.hy_right)

        precursors.hz_front_gpu = gpuarray.to_gpu(precursors.hz_front)
        precursors.hz_back_gpu = gpuarray.to_gpu(precursors.hz_back)
        precursors.hz_left_gpu = gpuarray.to_gpu(precursors.hz_left)
        precursors.hz_right_gpu = gpuarray.to_gpu(precursors.hz_right)


        mod = SourceModule(kernel_template_is.substitute(
            REAL = config.sim_config.dtypes['C_float_or_double'],
            NY_MATCOEFFS = self.updatecoeffsH.shape[1],
            NX_FIELDS = self.nx + 1,
            NY_FIELDS = self.ny + 1,
            NZ_FIELDS = self.nz + 1,
            NX_ID = self.ID.shape[1],
            NY_ID = self.ID.shape[2],
            NZ_ID = self.ID.shape[3]
        ))

        hsg_update_is_gpu = mod.get_function("hsg_update_is")

        bpg = (int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)) / 128)), 1, 1)

        # Bottom and Top
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwx), np.int32(self.nwy + 1),
            np.int32(1), # face
            np.int32(3), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Ex']),
            np.int32(precursors.hy_bottom.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ex_gpu.gpudata,
            precursors.hy_bottom_gpu.gpudata,
            precursors.hy_top_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwx + 1), np.int32(self.nwy),
            np.int32(1), # face
            np.int32(3), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Ey']),
            np.int32(precursors.hx_bottom.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ey_gpu.gpudata,
            precursors.hx_bottom_gpu.gpudata,
            precursors.hx_top_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Left and Right
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwy), np.int32(self.nwz + 1),
            np.int32(2), # face
            np.int32(1), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Ey']),
            np.int32(precursors.hz_left.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ey_gpu.gpudata,
            precursors.hz_left_gpu.gpudata,
            precursors.hz_right_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwy + 1), np.int32(self.nwz),
            np.int32(2), # face
            np.int32(1), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Ez']),
            np.int32(precursors.hy_left.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ez_gpu.gpudata,
            precursors.hy_left_gpu.gpudata,
            precursors.hy_right_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)
        
        # Front and Back
        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwx), np.int32(self.nwz + 1),
            np.int32(3), # face
            np.int32(2), # co
            np.int32(-1), np.int32(1),
            np.int32(self.IDlookup['Ex']),
            np.int32(precursors.hz_front.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ex_gpu.gpudata,
            precursors.hz_front_gpu.gpudata,
            precursors.hz_back_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_is_gpu(
            np.int32(self.nwx), np.int32(self.nwy), np.int32(self.nwz),
            np.int32(self.n_boundary_cells),
            np.int32(0), # offset
            np.int32(self.nwx + 1), np.int32(self.nwz),
            np.int32(3), # face
            np.int32(2), # co
            np.int32(1), np.int32(-1),
            np.int32(self.IDlookup['Ez']),
            np.int32(precursors.hx_front.shape[1]), # precursor node coefficient
            self.updatecoeffsE_gpu.gpudata,
            self.ID_gpu.gpudata,
            self.Ex_gpu.gpudata,
            precursors.hx_front_gpu.gpudata,
            precursors.hx_back_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        self.Ex = self.Ex_gpu.get()
        self.Ey = self.Ey_gpu.get()
        self.Ez = self.Ez_gpu.get()


        # Ex = c0(Ex) + c2(dHz) - c3(dHy)
        # Ey = c0(Ey) + c3(dHx) - c1(dHz)
        # Ez = c0(Ez) + c1(dHy) - c2(dHx)

        # bottom and top
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwx, self.nwy + 1, self.nwz, 1, self.Ex, precursors.hy_bottom, precursors.hy_top, self.IDlookup['Ex'], 1, -1, 3, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwx + 1, self.nwy, self.nwz, 1, self.Ey, precursors.hx_bottom, precursors.hx_top, self.IDlookup['Ey'], -1, 1, 3, config.get_model_config().ompthreads)

        # left and right
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwy, self.nwz + 1, self.nwx, 2, self.Ey, precursors.hz_left, precursors.hz_right, self.IDlookup['Ey'], 1, -1, 1, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwy + 1, self.nwz, self.nwx, 2, self.Ez, precursors.hy_left, precursors.hy_right, self.IDlookup['Ez'], -1, 1, 1, config.get_model_config().ompthreads)

        # front and back
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwx, self.nwz + 1, self.nwy, 3, self.Ex, precursors.hz_front, precursors.hz_back, self.IDlookup['Ex'], -1, 1, 2, config.get_model_config().ompthreads)
        # cython_update_is(self.nwx, self.nwy, self.nwz, self.updatecoeffsE, self.ID, self.n_boundary_cells, 0, self.nwx + 1, self.nwz, self.nwy, 3, self.Ez, precursors.hx_front, precursors.hx_back, self.IDlookup['Ez'], 1, -1, 2, config.get_model_config().ompthreads)

    def update_electric_os(self, main_grid):
        """

        """
        i_l = self.i0 - self.is_os_sep
        i_u = self.i1 + self.is_os_sep
        j_l = self.j0 - self.is_os_sep
        j_u = self.j1 + self.is_os_sep
        k_l = self.k0 - self.is_os_sep
        k_u = self.k1 + self.is_os_sep

        # Allocating GPU Arrays 

        main_grid.ID_gpu = gpuarray.to_gpu(main_grid.ID)
        main_grid.updatecoeffsE_gpu = gpuarray.to_gpu(main_grid.updatecoeffsE)
        
        main_grid.Ex_gpu = gpuarray.to_gpu(main_grid.Ex)
        main_grid.Ey_gpu = gpuarray.to_gpu(main_grid.Ey)
        main_grid.Ez_gpu = gpuarray.to_gpu(main_grid.Ez)
        
        self.Hx_gpu = gpuarray.to_gpu(self.Hx)
        self.Hy_gpu = gpuarray.to_gpu(self.Hy)
        self.Hz_gpu = gpuarray.to_gpu(self.Hz)

        mod = SourceModule(kernel_template_os.substitute(
            REAL = config.sim_config.dtypes['C_float_or_double'],
            NY_MATCOEFFS = main_grid.updatecoeffsH.shape[1],
            NX_FIELDS = main_grid.nx + 1,
            NY_FIELDS = main_grid.ny + 1,
            NZ_FIELDS = main_grid.nz + 1,
            NX_SUBFIELDS = self.nx + 1,
            NY_SUBFIELDS = self.ny + 1,
            NZ_SUBFIELDS = self.nz + 1,
            NX_ID = main_grid.ID.shape[1],
            NY_ID = main_grid.ID.shape[2],
            NZ_ID = main_grid.ID.shape[3]
        ))

        hsg_update_electric_os_gpu = mod.get_function("hsg_update_electric_os")

        bpg = (int(np.ceil(((main_grid.nx + 1) * (main_grid.ny + 1) * (main_grid.nz + 1)) / 128)), 1, 1)

        # Front and Back
        hsg_update_electric_os_gpu(
            np.int32(3), # face
            np.int32(2), # co
            np.int32(1), np.int32(-1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwy),
            np.int32(main_grid.IDlookup['Ex']),
            np.int32(i_l), np.int32(i_u),
            np.int32(k_l), np.int32(k_u + 1),
            np.int32(j_l), np.int32(j_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ex_gpu.gpudata,
            self.Hz_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_electric_os_gpu(
            np.int32(3), # face
            np.int32(2), # co
            np.int32(-1), np.int32(1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwy),
            np.int32(main_grid.IDlookup['Ez']),
            np.int32(i_l), np.int32(i_u + 1),
            np.int32(k_l), np.int32(k_u),
            np.int32(j_l), np.int32(j_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ez_gpu.gpudata,
            self.Hx_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Left and Right
        hsg_update_electric_os_gpu(
            np.int32(2), # face
            np.int32(1), # co
            np.int32(-1), np.int32(1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwx),
            np.int32(main_grid.IDlookup['Ey']),
            np.int32(j_l), np.int32(j_u ),
            np.int32(k_l), np.int32(k_u + 1),
            np.int32(i_l), np.int32(i_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ey_gpu.gpudata,
            self.Hz_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_electric_os_gpu(
            np.int32(2), # face
            np.int32(1), # co
            np.int32(1), np.int32(-1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwx),
            np.int32(main_grid.IDlookup['Ez']),
            np.int32(j_l), np.int32(j_u +1),
            np.int32(k_l), np.int32(k_u),
            np.int32(i_l), np.int32(i_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ez_gpu.gpudata,
            self.Hy_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Top and Bottom
        hsg_update_electric_os_gpu(
            np.int32(1), # face
            np.int32(3), # co
            np.int32(-1), np.int32(1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwz),
            np.int32(main_grid.IDlookup['Ex']),
            np.int32(i_l), np.int32(i_u),
            np.int32(j_l), np.int32(j_u + 1),
            np.int32(k_l), np.int32(k_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ex_gpu.gpudata,
            self.Hy_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_electric_os_gpu(
            np.int32(1), # face
            np.int32(3), # co
            np.int32(1), np.int32(-1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwz),
            np.int32(main_grid.IDlookup['Ey']),
            np.int32(i_l), np.int32(i_u + 1),
            np.int32(j_l), np.int32(j_u),
            np.int32(k_l), np.int32(k_u),
            main_grid.updatecoeffsE_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Ey_gpu.gpudata,
            self.Hx_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)


        main_grid.Ex = main_grid.Ex_gpu.get()
        main_grid.Ey = main_grid.Ey_gpu.get()
        main_grid.Ez = main_grid.Ez_gpu.get()


        # Args: sub_grid, normal, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup_id, field, inc_field, co, sign_n, sign_f

        # Form of FDTD update equations for E
        # Ex = c0(Ex) + c2(dHz) - c3(dHy)
        # Ey = c0(Ey) + c3(dHx) - c1(dHz)
        # Ez = c0(Ez) + c1(dHy) - c2(dHx)

        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 3, i_l, i_u, k_l, k_u + 1, j_l, j_u, self.nwy, main_grid.IDlookup['Ex'], main_grid.Ex, self.Hz, 2, 1, -1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)
        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 3, i_l, i_u + 1, k_l, k_u, j_l, j_u, self.nwy, main_grid.IDlookup['Ez'], main_grid.Ez, self.Hx, 2, -1, 1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

        # Left and Right
        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 2, j_l, j_u, k_l, k_u + 1, i_l, i_u, self.nwx, main_grid.IDlookup['Ey'], main_grid.Ey, self.Hz, 1, -1, 1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)
        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 2, j_l, j_u + 1, k_l, k_u, i_l, i_u, self.nwx, main_grid.IDlookup['Ez'], main_grid.Ez, self.Hy, 1, 1, -1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

        # Bottom and Top
        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 1, i_l, i_u, j_l, j_u + 1, k_l, k_u, self.nwz, main_grid.IDlookup['Ex'], main_grid.Ex, self.Hy, 3, -1, 1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)
        # cython_update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 1, i_l, i_u + 1, j_l, j_u, k_l, k_u, self.nwz, main_grid.IDlookup['Ey'], main_grid.Ey, self.Hx, 3, 1, -1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

    def update_magnetic_os(self, main_grid):
        """

        """
        i_l = self.i0 - self.is_os_sep
        i_u = self.i1 + self.is_os_sep
        j_l = self.j0 - self.is_os_sep
        j_u = self.j1 + self.is_os_sep
        k_l = self.k0 - self.is_os_sep
        k_u = self.k1 + self.is_os_sep

        # Form of FDTD update equations for H
        # Hz = c0Hz - c1Ey + c2Ex
        # Hy = c0Hy - c3Ex + c1Ez
        # Hx = c0Hx - c2Ez + c3Ey

        # Allocating GPU Arrays 

        main_grid.ID_gpu = gpuarray.to_gpu(main_grid.ID)
        main_grid.updatecoeffsH_gpu = gpuarray.to_gpu(main_grid.updatecoeffsH)
        
        main_grid.Hx_gpu = gpuarray.to_gpu(main_grid.Hx)
        main_grid.Hy_gpu = gpuarray.to_gpu(main_grid.Hy)
        main_grid.Hz_gpu = gpuarray.to_gpu(main_grid.Hz)
        
        self.Ex_gpu = gpuarray.to_gpu(self.Ex)
        self.Ey_gpu = gpuarray.to_gpu(self.Ey)
        self.Ez_gpu = gpuarray.to_gpu(self.Ez)

        mod = SourceModule(kernel_template_os.substitute(
            REAL = config.sim_config.dtypes['C_float_or_double'],
            NY_MATCOEFFS = main_grid.updatecoeffsH.shape[1],
            NX_FIELDS = main_grid.nx + 1,
            NY_FIELDS = main_grid.ny + 1,
            NZ_FIELDS = main_grid.nz + 1,
            NX_SUBFIELDS = self.nx + 1,
            NY_SUBFIELDS = self.ny + 1,
            NZ_SUBFIELDS = self.nz + 1,
            NX_ID = main_grid.ID.shape[1],
            NY_ID = main_grid.ID.shape[2],
            NZ_ID = main_grid.ID.shape[3]
        ))
        
        hsg_update_magnetic_os_gpu = mod.get_function("hsg_update_magnetic_os")

        bpg = (int(np.ceil(((main_grid.nx + 1) * (main_grid.ny + 1) * (main_grid.nz + 1)) / 128)), 1, 1)

        # Front and Back Faces
        hsg_update_magnetic_os_gpu(
            np.int32(3), # face
            np.int32(2), # co
            np.int32(1), np.int32(-1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwy),
            np.int32(main_grid.IDlookup['Hz']),
            np.int32(i_l), np.int32(i_u),
            np.int32(k_l), np.int32(k_u + 1),
            np.int32(j_l -1), np.int32(j_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hz_gpu.gpudata,
            self.Ex_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        hsg_update_magnetic_os_gpu(
            np.int32(3), # face
            np.int32(2), # co
            np.int32(-1), np.int32(1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwy),
            np.int32(main_grid.IDlookup['Hx']),
            np.int32(i_l), np.int32(i_u + 1),
            np.int32(k_l), np.int32(k_u),
            np.int32(j_l -1), np.int32(j_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hx_gpu.gpudata,
            self.Ez_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)

        # Left and Right Faces
        hsg_update_magnetic_os_gpu(
            np.int32(2), # face
            np.int32(1), # co
            np.int32(-1), np.int32(1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwx),
            np.int32(main_grid.IDlookup['Hz']),
            np.int32(j_l), np.int32(j_u),
            np.int32(k_l), np.int32(k_u + 1),
            np.int32(i_l -1), np.int32(i_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hz_gpu.gpudata,
            self.Ey_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)        

        hsg_update_magnetic_os_gpu(
            np.int32(2), # face
            np.int32(1), # co
            np.int32(1), np.int32(-1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwx),
            np.int32(main_grid.IDlookup['Hy']),
            np.int32(j_l), np.int32(j_u + 1),
            np.int32(k_l), np.int32(k_u),
            np.int32(i_l -1), np.int32(i_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hy_gpu.gpudata,
            self.Ez_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)      

          
        # Top and Bottom
        hsg_update_magnetic_os_gpu(
            np.int32(1), # face
            np.int32(3), # co
            np.int32(-1), np.int32(1),
            np.int32(1), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwz),
            np.int32(main_grid.IDlookup['Hy']),
            np.int32(i_l), np.int32(i_u),
            np.int32(j_l), np.int32(j_u + 1),
            np.int32(k_l -1), np.int32(k_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hy_gpu.gpudata,
            self.Ex_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)        

        hsg_update_magnetic_os_gpu(
            np.int32(1), # face
            np.int32(3), # co
            np.int32(1), np.int32(-1),
            np.int32(0), # mid
            np.int32(self.ratio),
            np.int32(self.is_os_sep),
            np.int32(self.n_boundary_cells),
            np.int32(self.nwz),
            np.int32(main_grid.IDlookup['Hx']),
            np.int32(i_l), np.int32(i_u + 1),
            np.int32(j_l), np.int32(j_u),
            np.int32(k_l -1), np.int32(k_u),
            main_grid.updatecoeffsH_gpu.gpudata,
            main_grid.ID_gpu.gpudata,
            main_grid.Hx_gpu.gpudata,
            self.Ey_gpu.gpudata,
            block = (128,1,1),
            grid = bpg)        


        # Copying the data back to host
        main_grid.Hx = main_grid.Hx_gpu.get()
        main_grid.Hy = main_grid.Hy_gpu.get()
        main_grid.Hz = main_grid.Hz_gpu.get()
        
        # Args: sub_grid, normal, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup_id, field, inc_field, co, sign_n, sign_f):
        # Front and back
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 3, i_l, i_u, k_l, k_u + 1, j_l - 1, j_u, self.nwy, main_grid.IDlookup['Hz'], main_grid.Hz, self.Ex, 2, 1, -1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads) 
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 3, i_l, i_u + 1, k_l, k_u, j_l - 1, j_u, self.nwy, main_grid.IDlookup['Hx'], main_grid.Hx, self.Ez, 2, -1, 1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

        # Left and Right
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 2, j_l, j_u, k_l, k_u + 1, i_l - 1, i_u, self.nwx, main_grid.IDlookup['Hz'], main_grid.Hz, self.Ey, 1, -1, 1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)
        
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 2, j_l, j_u + 1, k_l, k_u, i_l - 1, i_u, self.nwx, main_grid.IDlookup['Hy'], main_grid.Hy, self.Ez, 1, 1, -1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

        # bottom and top
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 1, i_l, i_u, j_l, j_u + 1, k_l - 1, k_u, self.nwz, main_grid.IDlookup['Hy'], main_grid.Hy, self.Ex, 3, -1, 1, 1, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)
        # cython_update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 1, i_l, i_u + 1, j_l, j_u, k_l - 1, k_u, self.nwz, main_grid.IDlookup['Hx'], main_grid.Hx, self.Ey, 3, 1, -1, 0, self.ratio, self.is_os_sep, self.n_boundary_cells, config.get_model_config().ompthreads)

    def print_info(self):
        """Print information about the subgrid."""
        
        logger.info('')
        logger.info(f'[{self.name}] Type: {self.gridtype}')
        logger.info(f'[{self.name}] Ratio: 1:{self.ratio}')
        logger.info(f'[{self.name}] Spatial discretisation: {self.dx:g} x {self.dy:g} x {self.dz:g}m')
        logger.info(f'[{self.name}] Extent: {self.i0 * self.dx * self.ratio}m, {self.j0 * self.dy * self.ratio}m, {self.k0 * self.dz * self.ratio}m to {self.i1 * self.dx * self.ratio}m, {self.j1 * self.dy * self.ratio}m, {self.k1 * self.dz * self.ratio}m')
        logger.debug(f'[{self.name}] Working region: {self.nwx} x {self.nwy} x {self.nwz} cells')
        logger.debug(f'[{self.name}] Total region: {self.nx:d} x {self.ny:d} x {self.nz:d} = {(self.nx * self.ny * self.nz):g} cells')
        # Total region = working region + 2 * (is_os_sep * pml_separation * pml_thickness)
        # is_os_sep - number of main grid cells between the Inner Surface and 
        #               the Outer Surface. Defaults to 3. Multiply by ratio to 
        #               get sub-grid cells.
        # pml_separation - number of sub-grid cells between the Outer Surface 
        #                   and the PML. Defaults to ratio // 2 + 2.
        # pml_thickness - number of PML cells on each of the 6 sides of the 
        #                   sub-grid. Defaults to 6.
        logger.info(f'[{self.name}] Time step (at CFL limit): {self.dt:g} secs')
