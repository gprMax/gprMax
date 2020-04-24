# Copyright (C) 2015-2020: The University of Edinburgh
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

# Lists to hold material and sources/receiver identifiers written in VTK file in tags <gprMax3D> <Material> and <gprMax3D> <Sources/Receivers>
materials = []
srcs_pml = []
rxs = []
with open(model.FileName[0], 'rb') as f:
    for line in f:
        if line.startswith(b'<Material'):
            line.rstrip('\n')
            tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            materials.append(tmp)
        elif line.startswith(b'<Sources') or line.startswith(b'<PML'):
            line.rstrip('\n')
            tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            srcs_pml.append(tmp)
        elif line.startswith(b'<Receivers'):
            line.rstrip('\n')
            tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
            rxs.append(tmp)

if materials:
    # Get range of data
    mat_datarange = model.CellData.GetArray('Material').GetRange()

    # Create threshold for materials (name and numeric value)
    for x in range(0, int(mat_datarange[1]) + 1):
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

if srcs_pml:
    # Get ranges of data
    srcs_pml_datarange = model.CellData.GetArray('Sources_PML').GetRange()

    # Create threshold for sources/pml (name and numeric value)
    for x in range(1, int(srcs_pml_datarange[1]) + 1):
        for y in range(len(srcs_pml)):
            if srcs_pml[y][0] == x:
                threshold = Threshold(Input=model)
                threshold.Scalars = 'Sources_PML'
                threshold.ThresholdRange = [srcs_pml[y][0], srcs_pml[y][0]]

                RenameSource(srcs_pml[y][1], threshold)

                # Show data in view
                thresholddisplay = Show(threshold, renderview)
                thresholddisplay.ColorArrayName = 'Sources_PML'

                if srcs_pml[y][0] == 1:
                    thresholddisplay.Opacity = 0.5

if rxs:
    # Get ranges of data
    rxs_datarange = model.CellData.GetArray('Receivers').GetRange()

    # Create threshold for sources/pml (name and numeric value)
    for x in range(1, int(rxs_datarange[1]) + 1):
        for y in range(len(rxs)):
            if rxs[y][0] == x:
                threshold = Threshold(Input=model)
                threshold.Scalars = 'Receivers'
                threshold.ThresholdRange = [rxs[y][0], rxs[y][0]]

                RenameSource(rxs[y][1], threshold)

                # Show data in view
                thresholddisplay = Show(threshold, renderview)
                thresholddisplay.ColorArrayName = 'Receivers'

# renderview.CameraParallelProjection = 1
RenderAllViews()

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)
