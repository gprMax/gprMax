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
    pml_gp = AppendDatasets(Input=tmp)
    RenameSource('PML - All', pml_gp)
    pml_view = Show(pml_gp)
    pml_view.Opacity = 0.5


# Get whatever source is loaded (should be model)
model = GetActiveSource()

# Get active view
renderview = GetActiveView()

# Show Data Axes Grid
renderview.AxesGrid.Visibility = 1

# Hide display of root data
Hide(model)

# VTI or VTP file
if len(model.FileName) == 1:
    files = model.FileName
    dirname = os.path.dirname(files[0])
# PVD file
else:
    files = []
    dirname = os.path.dirname(model.FileName)
    tree = ET.parse(model.FileName)
    root = tree.getroot()
    for elem in root:
        for subelem in elem.findall('DataSet'):
            tmp = os.path.join(dirname, subelem.get('file').encode('utf-8'))
            files.append(tmp)

materials = {}
srcs = {}
srcs_old = {}
rxs = {}
rxs_old = {}
pmls_old = {}
material_ID_max = 0

for file in files:
    with open(file, 'rb') as f:
        # Get max numID of materials
        if model.CellData.GetArray('Material').GetRange()[1] > material_ID_max:
            material_ID_max = int(model.CellData.GetArray('Material').GetRange()[1])

        # Read gprMax XML section
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        xml_pos = mm.find('<gprMax')
        mm.seek(xml_pos)
        xml = mm.read(mm.size() - xml_pos)
        root = ET.fromstring(xml)

        # Print gprMax XML section that has been read for debugging
        # ET.dump(root)

        # New behaviour - gprMax >v.4
        try:
            root.attrib['Version']
            v4 = True
            dx_dy_dz = tuple(float(s) for s in root.get(
                'dx_dy_dz').strip('()').split(','))
            nx_ny_nz = tuple(float(s) for s in root.get(
                'nx_ny_nz').strip('()').split(','))
            if any(x == 1 for x in nx_ny_nz):
                renderview.CameraParallelProjection = 1
            try: 
                root.attrib['PMLthickness']
                pmlthick = tuple(int(s) for s in root.get(
                    'PMLthickness').strip('[]').split(','))
            except:
                pass

        # Old behaviour - gprMax <v.4
        except:
            v4 = False

        if v4:
            # Read material names and numeric IDs into a dict
            for elem in root:
                for subelem in elem.findall('Material'):
                    materials[subelem.get('ID')] = int(subelem.get('numID'))

            # Read sources
            for elem in root:
                for subelem in elem.findall('Source'):
                    pos = tuple(float(s) for s in subelem.get(
                        'position').strip('()').split(','))
                    srcs[subelem.get('name')] = pos

            # Read receivers
            for elem in root:
                for subelem in elem.findall('Receiver'):
                    pos = tuple(float(s) for s in subelem.get(
                        'position').strip('()').split(','))
                    rxs[subelem.get('name')] = pos

        else:
            # Read material names and numeric IDs into a dict
            for elem in root.findall('Material'):
                materials[elem.get('name')] = int(elem.text)

            # Read sources
            for elem in root.findall('Sources'):
                srcs_old[elem.get('name')] = int(elem.text)

            # Read receivers
            for elem in root.findall('Receivers'):
                rxs_old[elem.get('name')] = int(elem.text)

            # Read PMLs
            for elem in root.findall('PML'):
                pmls_old[elem.get('name')] = int(elem.text)

# Create a Threshold (filter) for each material
material_range = range(0, material_ID_max + 1)
for k, v in sorted(materials.items(), key=lambda x: x[1]):
    if v in material_range:
        threshold = Threshold(Input=model)
        threshold.Scalars = 'Material'
        threshold.ThresholdRange = [v, v]
        RenameSource(k, threshold)

        # Show data in view, except for free_space
        if v != 1:
            thresholddisplay = Show(threshold, renderview)
            thresholddisplay.ColorArrayName = 'Material'

# New behaviour
if v4:
    # Display sources and receivers as Paraview box sources
    for k, v in srcs.items() + rxs.items():
        src_rx = Box(Center=[v[0] + dx_dy_dz[0]/2,
                            v[1] + dx_dy_dz[1]/2,
                            v[2] + dx_dy_dz[2]/2],
                    XLength=dx_dy_dz[0], YLength=dx_dy_dz[1], ZLength=dx_dy_dz[2])
        RenameSource(k, src_rx)
        Show(src_rx)

    # Display PMLs as Paraview box sources
    try:
        pmlthick
        display_pmls_new(pmlthick, dx_dy_dz, nx_ny_nz)
    except:
        pass
    
# Old behaviour
else:
    # Create threshold for sources/pml(name and numeric value)
    srcs_pmls_old = dict(srcs_old.items() + pmls_old.items())
    if srcs_pmls_old:
        for k, v in srcs_pmls_old.items():
            threshold = Threshold(Input=model)
            threshold.Scalars = 'Sources_PML'
            threshold.ThresholdRange = [v, v]
            RenameSource(k, threshold)

            # Show data in view
            thresholddisplay = Show(threshold, renderview)
            thresholddisplay.ColorArrayName = 'Sources_PML'

            if v == 1:
                thresholddisplay.Opacity = 0.5

    if rxs_old:
        # Create threshold for receivers (name and numeric value)
        for k, v in rxs_old.items():
            threshold = Threshold(Input=model)
            threshold.Scalars = 'Receivers'
            threshold.ThresholdRange = [v, v]
            RenameSource(k, threshold)

            # Show data in view
            thresholddisplay = Show(threshold, renderview)
            thresholddisplay.ColorArrayName = 'Receivers'

RenderAllViews()

# Reset view to fit data
renderview.ResetCamera()

# Show color bar/color legend
# thresholdDisplay.SetScalarBarVisibility(renderview, False)
