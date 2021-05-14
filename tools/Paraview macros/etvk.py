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

import mmap
import os
from xml.etree import ElementTree as ET
import json

def display_pmls_new(pmlthick, dx_dy_dz, nx_ny_nz):
    """Display PMLs as box sources using PML thickness values.
        Only suitable for gprMax > v.4

    Args:
        pmlthick (tuple): PML thickness values for each slab (cells)
        dx_dy_dz (tuple): Spatial resolution (m)
        nx_ny_dz (tuple): Domain size (cells)
    """

    pml_names = ['x0', 'y0', 'z0', 'xmax', 'ymax', 'zmax']
    pmls = dict.fromkeys(pml_names, None)

    if pmlthick[0] != 0:
        x0 = Box(Center=[pmlthick[0] * dx_dy_dz[0] / 2,
                         nx_ny_nz[1] * dx_dy_dz[1] / 2,
                         nx_ny_nz[2] * dx_dy_dz[2] / 2],
                 XLength=pmlthick[0] * dx_dy_dz[0],
                 YLength=nx_ny_nz[1] * dx_dy_dz[1],
                 ZLength=nx_ny_nz[2] * dx_dy_dz[2])
        pmls['x0'] = x0

    if pmlthick[3] != 0:
        xmax = Box(Center=[dx_dy_dz[0] * (nx_ny_nz[0] - pmlthick[3] / 2),
                           nx_ny_nz[1] * dx_dy_dz[1] / 2,
                           nx_ny_nz[2] * dx_dy_dz[2] / 2],
                   XLength=pmlthick[3] * dx_dy_dz[0],
                   YLength=nx_ny_nz[1] * dx_dy_dz[1],
                   ZLength=nx_ny_nz[2] * dx_dy_dz[2])
        pmls['xmax'] = xmax

    if pmlthick[1] != 0:
        y0 = Box(Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2,
                         pmlthick[1] * dx_dy_dz[1] / 2,
                         nx_ny_nz[2] * dx_dy_dz[2] / 2],
                 XLength=nx_ny_nz[0] * dx_dy_dz[0],
                 YLength=pmlthick[1] * dx_dy_dz[1],
                 ZLength=nx_ny_nz[2] * dx_dy_dz[2])
        pmls['y0'] = y0

    if pmlthick[4] != 0:
        ymax = Box(Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2,
                           dx_dy_dz[1] * (nx_ny_nz[1] - pmlthick[4] / 2),
                           nx_ny_nz[2] * dx_dy_dz[2] / 2],
                   XLength=nx_ny_nz[0] * dx_dy_dz[0],
                   YLength=pmlthick[4] * dx_dy_dz[1],
                   ZLength=nx_ny_nz[2] * dx_dy_dz[2])
        pmls['ymax'] = ymax

    if pmlthick[2] != 0:
        z0 = Box(Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2,
                         nx_ny_nz[1] * dx_dy_dz[1] / 2,
                         pmlthick[2] * dx_dy_dz[2] / 2],
                 XLength=nx_ny_nz[0] * dx_dy_dz[0],
                 YLength=nx_ny_nz[1] * dx_dy_dz[1],
                 ZLength=pmlthick[2] * dx_dy_dz[2])
        pmls['z0'] = z0

    if pmlthick[5] != 0:
        zmax = Box(Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2,
                           nx_ny_nz[1] * dx_dy_dz[1] / 2,
                           dx_dy_dz[2] * (nx_ny_nz[2] - pmlthick[5] / 2)],
                   XLength=nx_ny_nz[0] * dx_dy_dz[0],
                   YLength=nx_ny_nz[1] * dx_dy_dz[1],
                   ZLength=pmlthick[5] * dx_dy_dz[2])
        pmls['zmax'] = zmax

    # Name PML sources and set opacity
    tmp = []
    for pml in pmls:
        if pmls[pml]:
            RenameSource('PML - ' + pml, pmls[pml])
            Hide(pmls[pml], renderview)
            tmp.append(pmls[pml])

    # Create a group of PMLs to switch on/off easily
    if tmp:
        pml_gp = AppendDatasets(Input=tmp)
        RenameSource('PML - All', pml_gp)
        pml_view = Show(pml_gp)
        pml_view.Opacity = 0.5

def load_src_rx(srcs, dl):
    # Display sources and receivers as Paraview box sources
    for item in srcs:
        p = item['position']
        n = item['name']
        src_rx = Box(Center=[p[0] + dl[0]/2,
                             p[1] + dl[1]/2,
                             p[2] + dl[2]/2],
                     XLength=dl[0], YLength=dl[1], ZLength=dl[2])
        RenameSource(n, src_rx)
        Show(src_rx)


def load_materials(ml):
    # create a threshold filter for each material type
    for i, m in enumerate(ml):
        threshold = Threshold(Input=model)
        threshold.ThresholdRange = [i, i]
        threshold.Scalars = ['CELLS', 'Material']
        RenameSource(m, threshold)

        # Show data in view, except for free_space
        if i != 1:
            thresholddisplay = Show(threshold, renderview)
        thresholddisplay.ColorArrayName = ['CELLS', 'Material']
        threshold.UpdatePipeline()

from paraview.simple import *

# Get whatever source is loaded (should be model)
model = GetActiveSource()

# Get active view
renderview = GetActiveView()

# Show Data Axes Grid
renderview.AxesGrid.Visibility = 1

# Hide display of root data
Hide(model)

# one file only for now

fp = model.FileName[0]

# materials string
ms = None

# load json embedded in comment of line 3
with open(fp, 'rb') as f:
    f.readline()
    f.readline()
    # comments
    c = f.readline().decode()

# strip comment tags
c = c[5:-5]
# model information
c = json.loads(c)

# discretisation
dl = c['dx_dy_dz']
# n voxels
nl = c['nx_ny_nz']

# load material filters
try:
    ml = c['Materials']
    load_materials(ml)
except IndexError:
    print('No Materials to load')

# load sources
try:
    srcs = c['Sources']
    load_src_rx(srcs, dl)
except IndexError:
    print('No sources to load}')

# load rxs
try:
    rxs = c['Receivers']
    load_src_rx(rxs, dl)
except IndexError:
    print('No receivers to load')

# load pmls
try:
    pt = c['PMLthickness']
    display_pmls_new(pt, dl, nl)
except IndexError:
    print('No PMLs to load')

RenderAllViews()

# Reset view to fit data
renderview.ResetCamera()

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)
