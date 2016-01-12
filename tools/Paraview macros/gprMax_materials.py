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

# Get active view
renderview = GetActiveView()

# Show data in view
Show(model, renderview)

# Reset view to fit data
renderview.ResetCamera()

# List to hold material identifiers written in VTK file in tags <gprMax3D> <Material>
materials = []
with open(model.FileName[0], 'rb') as f:
    for line in f:
        if line.startswith('<Material'):
            line.rstrip('\n')
            tmp = (float(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            materials.append(tmp)

# If the geometry file does not contain any information on materials names
if not materials:
    # Get range of data and set upper bound
    datarange = model.CellData.GetArray(0).GetRange()
    
    for x in range(0, int(datarange[1]) + 1):
        threshold = Threshold(Input=model)
        threshold.ThresholdRange = [x, x]
        
        if x != 1:
            # Turn on show for all materials except free_space
            thresholdDisplay = Show(threshold, renderview)
        
        # Name materials
        if x == 0:
            RenameSource('pec', threshold)
        elif x == 1:
            RenameSource('free_space', threshold)
        else:
            RenameSource('material ' + str(x), threshold)

else:
    # Create threshold for materials (name and numeric value)
    for material in range(len(materials)):
        threshold = Threshold(Input=model)
        threshold.ThresholdRange = [materials[material][0], materials[material][0]]
        
        RenameSource(materials[material][1], threshold)
        
        if materials[material][0] != 1:
            # Show data in view
            thresholdDisplay = Show(threshold, renderview)

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)