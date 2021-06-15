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



def load_materials(ml):
    # create a threshold filter for each material type
    for i, m in enumerate(ml):
        threshold = Threshold(Input=model)
        threshold.ThresholdRange = [i, i]
        print(threshold.Scalars)
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
except KeyError:
    print('No Materials to load')

RenderAllViews()

# Reset view to fit data
renderview.ResetCamera()

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)
