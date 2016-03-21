# Copyright (C) 2015-2016: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from paraview.simple import *
from xml.etree import ElementTree as ET

# Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Get whatever source is loaded (should be model)
model = GetActiveSource()

# Get display properties
dp = GetDisplayProperties(model)

# Get active view
renderview = GetActiveView()

# Show data in view
Show(model, renderview)

# Reset view to fit data
renderview.ResetCamera()

# Lists to hold material and sources/receiver identifiers written in VTK file in tags <gprMax3D> <Material> and <gprMax3D> <Sources/Receivers>
materials = []
srcsrxs = []
with open(model.FileName[0], 'r') as f:
    for line in f:
        if line.startswith('<Material'):
            line.rstrip('\n')
            tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            materials.append(tmp)
        elif line.startswith('<Sources'):
            line.rstrip('\n')
            tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            srcsrxs.append(tmp)

# Get ranges of data
matdatarange = model.CellData.GetArray(0).GetRange()
srcrxdatarange = model.CellData.GetArray(1).GetRange()

if materials:
    # Create threshold for materials (name and numeric value)
    for x in range(0, int(matdatarange[1]) + 1):
        for y in range(len(materials)):
            if materials[y][0] == x:
                threshold = Threshold(Input=model)
                threshold.Scalars = 'Material'
                threshold.ThresholdRange = [materials[y][0], materials[y][0]]
                
                RenameSource(materials[y][1], threshold)
                
                if materials[y][0] != 1:
                    # Show data in view
                    thresholddisplay = Show(threshold, renderview)
                    thresholddisplay.ColorArrayName = 'Material'

if srcsrxs:
    # Create threshold for sources/receivers (name and numeric value)
    for x in range(0, int(srcrxdatarange[1]) + 1):
        for y in range(len(srcsrxs)):
            if srcsrxs[y][0] == x:
                threshold = Threshold(Input=model)
                threshold.Scalars = 'Sources_Receivers'
                threshold.ThresholdRange = [srcsrxs[y][0], srcsrxs[y][0]]
                
                RenameSource(srcsrxs[y][1], threshold)
                
                # Show data in view
                thresholddisplay = Show(threshold, renderview)
                thresholddisplay.ColorArrayName = 'Sources_Receivers'

Render()
# Show color bar/color legend
#thresholdDisplay.SetScalarBarVisibility(renderview, False)