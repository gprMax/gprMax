# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

import json
import os

from paraview.simple import (
    AppendDatasets,
    Box,
    GetActiveSource,
    GetActiveView,
    GetParaViewVersion,
    Hide,
    OpenDataFile,
    RenameSource,
    RenderAllViews,
    SetActiveSource,
    Show,
    Threshold,
)


def threshold_filt(input, lt, ut, scalars):
    """Create threshold filter according to Paraview version.

    Args:
        input (array): input data to threshold filter
        lt, ut (int): lower and upper bounds of thresholding operation
        scalars (list/str): name of scalar array to perform thresholding

    Returns:
        threshold (object): threshold filter
    """

    # Read Paraview version number to set threshold filter method
    pvv = GetParaViewVersion()

    threshold = Threshold(Input=input)
    threshold.Scalars = scalars

    if pvv.major == 5 and pvv.minor < 10:
        threshold.ThresholdRange = [lt, ut]
    else:
        threshold.LowerThreshold = lt
        threshold.UpperThreshold = ut

    return threshold


def display_pmls(pmlthick, dx_dy_dz, nx_ny_nz):
    """Display PMLs as box sources using PML thickness values.
        Only suitable for gprMax >= v4

    Args:
        pmlthick (tuple): PML thickness values for each slab (cells)
        dx_dy_dz (tuple): Spatial resolution (m)
        nx_ny_dz (tuple): Domain size (cells)
    """

    pml_names = ["x0", "y0", "z0", "xmax", "ymax", "zmax"]
    pmls = dict.fromkeys(pml_names, None)
    SetActiveSource(pv_data)

    if pmlthick[0] != 0:
        x0 = Box(
            Center=[pmlthick[0] * dx_dy_dz[0] / 2, nx_ny_nz[1] * dx_dy_dz[1] / 2, nx_ny_nz[2] * dx_dy_dz[2] / 2],
            XLength=pmlthick[0] * dx_dy_dz[0],
            YLength=nx_ny_nz[1] * dx_dy_dz[1],
            ZLength=nx_ny_nz[2] * dx_dy_dz[2],
        )
        pmls["x0"] = x0

    if pmlthick[3] != 0:
        xmax = Box(
            Center=[
                dx_dy_dz[0] * (nx_ny_nz[0] - pmlthick[3] / 2),
                nx_ny_nz[1] * dx_dy_dz[1] / 2,
                nx_ny_nz[2] * dx_dy_dz[2] / 2,
            ],
            XLength=pmlthick[3] * dx_dy_dz[0],
            YLength=nx_ny_nz[1] * dx_dy_dz[1],
            ZLength=nx_ny_nz[2] * dx_dy_dz[2],
        )
        pmls["xmax"] = xmax

    if pmlthick[1] != 0:
        y0 = Box(
            Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2, pmlthick[1] * dx_dy_dz[1] / 2, nx_ny_nz[2] * dx_dy_dz[2] / 2],
            XLength=nx_ny_nz[0] * dx_dy_dz[0],
            YLength=pmlthick[1] * dx_dy_dz[1],
            ZLength=nx_ny_nz[2] * dx_dy_dz[2],
        )
        pmls["y0"] = y0

    if pmlthick[4] != 0:
        ymax = Box(
            Center=[
                nx_ny_nz[0] * dx_dy_dz[0] / 2,
                dx_dy_dz[1] * (nx_ny_nz[1] - pmlthick[4] / 2),
                nx_ny_nz[2] * dx_dy_dz[2] / 2,
            ],
            XLength=nx_ny_nz[0] * dx_dy_dz[0],
            YLength=pmlthick[4] * dx_dy_dz[1],
            ZLength=nx_ny_nz[2] * dx_dy_dz[2],
        )
        pmls["ymax"] = ymax

    if pmlthick[2] != 0:
        z0 = Box(
            Center=[nx_ny_nz[0] * dx_dy_dz[0] / 2, nx_ny_nz[1] * dx_dy_dz[1] / 2, pmlthick[2] * dx_dy_dz[2] / 2],
            XLength=nx_ny_nz[0] * dx_dy_dz[0],
            YLength=nx_ny_nz[1] * dx_dy_dz[1],
            ZLength=pmlthick[2] * dx_dy_dz[2],
        )
        pmls["z0"] = z0

    if pmlthick[5] != 0:
        zmax = Box(
            Center=[
                nx_ny_nz[0] * dx_dy_dz[0] / 2,
                nx_ny_nz[1] * dx_dy_dz[1] / 2,
                dx_dy_dz[2] * (nx_ny_nz[2] - pmlthick[5] / 2),
            ],
            XLength=nx_ny_nz[0] * dx_dy_dz[0],
            YLength=nx_ny_nz[1] * dx_dy_dz[1],
            ZLength=pmlthick[5] * dx_dy_dz[2],
        )
        pmls["zmax"] = zmax

    # Name PML sources and set opacity
    tmp = []
    for pml in pmls:
        if pmls[pml]:
            RenameSource("PML - " + pml, pmls[pml])
            Hide(pmls[pml], pv_view)
            tmp.append(pmls[pml])

    # Create a group of PMLs to switch on/off easily
    if tmp:
        pml_gp = AppendDatasets(Input=tmp)
        RenameSource("PML - All", pml_gp)
        pml_view = Show(pml_gp)
        pml_view.Opacity = 0.5


# Get whatever source is loaded - should be loaded file (.vt*) or files (.pvd)
pv_data = GetActiveSource()

# Hide display of root data
Hide(pv_data)

# Single .vti or .vtu file
file = pv_data.FileName[0]

# Read and display data from file, i.e. materials, sources,  receivers, and PMLs
with open(file, "rb") as f:
    # Comments () embedded in line 3 of file
    f.readline()
    f.readline()
    c = f.readline().decode()
    # Strip comment tags
    c = c[5:-5]
    # Model information
    c = json.loads(c)
    print("\ngprMax version: " + c["gprMax_version"])
    print(file)

################
# Display data #
################
pv_view = GetActiveView()
pv_view.AxesGrid.Visibility = 1  # Show Data Axes Grid
pv_disp = Show(pv_data, pv_view)
pv_disp.ColorArrayName = ["CELLS", "Material"]

# Discretisation
dl = c["dx_dy_dz"]
# Number of voxels
nl = c["nx_ny_nz"]

# Materials
try:
    for i, mat in enumerate(c["Materials"]):
        threshold = threshold_filt(pv_data, i, i, ["CELLS", "Material"])
        RenameSource(mat, threshold)

        # Show data in view, except for free_space
        if i != 1:
            thresholddisplay = Show(threshold, pv_view)
            thresholddisplay.ColorArrayName = ["CELLS", "Material"]
        threshold.UpdatePipeline()
except KeyError:
    print("No materials to load")

# Display any sources
try:
    for item in c["Sources"]:
        pos = item["position"]
        name = item["name"]
        src = Box(
            Center=[pos[0] + dl[0] / 2, pos[1] + dl[1] / 2, pos[2] + dl[2] / 2],
            XLength=dl[0],
            YLength=dl[1],
            ZLength=dl[2],
        )
        RenameSource(name, src)
        Show(src)
except KeyError:
    print("No sources to load")

# Display any receivers
try:
    for item in c["Receivers"]:
        pos = item["position"]
        name = item["name"]
        rx = Box(
            Center=[pos[0] + dl[0] / 2, pos[1] + dl[1] / 2, pos[2] + dl[2] / 2],
            XLength=dl[0],
            YLength=dl[1],
            ZLength=dl[2],
        )
        RenameSource(name, rx)
        Show(rx)
except KeyError:
    print("No receivers to load")

# Display any PMLs
try:
    pt = c["PMLthickness"]
    display_pmls(pt, dl, nl)
except KeyError:
    print("No PMLs to load")


RenderAllViews()

# Reset view to fit data
pv_view.ResetCamera()
