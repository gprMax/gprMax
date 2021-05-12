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

# Get max numID of materials
fp = model.FileName[0]

# materials string
ms = None

# read the material list comment from the file
b = True
with open(fp, 'rb') as f:
    while(b):
        line = f.readline().decode()
        if 'pec,free_space' in line:
            # materials string
            ms = line
            b = False

if ms:
    # materials list
    ml = ms[5:-5].split(',')

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

# create a threshold filter for the pml
threshold = Threshold(Input=model)
threshold.ThresholdRange = [1, 1]
threshold.Scalars = ['CELLS', 'pml']
RenameSource('pml', threshold)

# Show data in view, except for free_space
thresholddisplay = Show(threshold, renderview)
thresholddisplay.ColorArrayName = ['CELLS', 'pml']
thresholddisplay.Opacity = 0.5
threshold.UpdatePipeline()

RenderAllViews()

# Reset view to fit data
renderview.ResetCamera()

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)
