# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

from paraview.servermanager import Proxy, SourceProxy
from paraview.simple import (
    AppendDatasets,
    Box,
    ColorBy,
    FetchData,
    GetActiveSource,
    GetActiveView,
    GetDisplayProperties,
    GetParaViewVersion,
    Hide,
    RenameSource,
    RenderAllViews,
    SetActiveSource,
    Show,
    Threshold,
)
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonCore import vtkStringArray

COLOUR_SCALARS = ("CELLS", "Material")


def threshold_materials(source: SourceProxy, view: Proxy, material_ids: vtkStringArray):
    """Create threshold filter according to Paraview version.

    Args:
        source: Input to the threshold filter.
        view: The view proxy to show the threshold in.
        material_ids: Array of material ids. A new threshold filter will
            be created for each material.
    """

    # Read Paraview version number to set threshold filter method
    pvv = GetParaViewVersion()

    for index in range(material_ids.GetNumberOfValues()):
        threshold = Threshold(Input=source, Scalars=COLOUR_SCALARS)

        if pvv.major == 5 and pvv.minor < 10:
            threshold.ThresholdRange = [index, index]
        else:
            threshold.LowerThreshold = index
            threshold.UpperThreshold = index

        RenameSource(material_ids.GetValue(index), threshold)

        # Show data in view, except for free_space
        if index != 1:
            Show(threshold, view)

        threshold.UpdatePipeline()


def create_box_sources(names: vtkStringArray, positions: dsa.VTKArray, dl: dsa.VTKArray):
    """Create new single cell box sources.

    Args:
        names: Array of N names for the new sources.
        positions: x, y, z coordinates of the new sources. This should
            have shape (N, 3).
        dl: x, y, z spatial resolution.
    """
    for index in range(names.GetNumberOfValues()):
        name = names.GetValue(index)
        pos = positions[index]
        src = Box(
            Center=pos + dl / 2,
            XLength=dl[0],
            YLength=dl[1],
            ZLength=dl[2],
        )
        RenameSource(name, src)
        Show(src)


def display_pmls(pmlthick: dsa.VTKArray, dx_dy_dz: dsa.VTKArray, nx_ny_nz: dsa.VTKArray):
    """Display PMLs as box sources using PML thickness values.

    Only suitable for gprMax >= v4.

    Args:
        pmlthick: PML thickness values for each slab (cells).
        dx_dy_dz: Spatial resolution (m).
        nx_ny_dz: Domain size (cells).
    """

    pml_names = ["x0", "y0", "z0", "xmax", "ymax", "zmax"]
    pmls = dict.fromkeys(pml_names, None)

    if pmlthick[0] != 0:
        x0 = Box(
            Center=[
                pmlthick[0] * dx_dy_dz[0] / 2,
                nx_ny_nz[1] * dx_dy_dz[1] / 2,
                nx_ny_nz[2] * dx_dy_dz[2] / 2,
            ],
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
            Center=[
                nx_ny_nz[0] * dx_dy_dz[0] / 2,
                pmlthick[1] * dx_dy_dz[1] / 2,
                nx_ny_nz[2] * dx_dy_dz[2] / 2,
            ],
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
            Center=[
                nx_ny_nz[0] * dx_dy_dz[0] / 2,
                nx_ny_nz[1] * dx_dy_dz[1] / 2,
                pmlthick[2] * dx_dy_dz[2] / 2,
            ],
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


def is_valid_key(key: str, dictionary: dict) -> bool:
    """Check key exists and is not an empty array (i.e. VTKNoneArray).

    Args:
        key: Name of key to check.
        dictionary: Dictionary like object that should contain the key.

    Returns:
        is_valid_key: True if the key is in the dictionary and is not
            of type VTKNoneArray. False otherwise.
    """
    return key in dictionary.keys() and not isinstance(dictionary[key], dsa.VTKNoneArray)


class HaltException(Exception):
    pass


try:
    print("=============== Running gprMax macro ===============", end="\n\n")

    # Get active source - should be loaded file (.vtkhdf)
    pv_source = GetActiveSource()

    if pv_source is None:
        raise HaltException("ERROR: No active source.")

    pv_source.UpdatePipeline()

    raw_data = FetchData(pv_source)[0]
    data = dsa.WrapDataObject(raw_data)

    # Check the necessary metadata is attached to the file
    metadata_keys = data.FieldData.keys()
    compulsory_keys = ["gprMax_version", "nx_ny_nz", "dx_dy_dz", "material_ids"]
    missing_keys = []
    for key in compulsory_keys:
        if key not in metadata_keys:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        keys = "\n- ".join(missing_keys)
        print(f"Missing metadata fields: \n- {keys}\n")
        if len(metadata_keys) > 0:
            keys = "\n- ".join(metadata_keys)
            print(f"Found:  \n- {keys}\n")
        raise HaltException(
            "ERROR: Missing gprMax metadata information. Do you have the correct source selected?"
        )

    # gprMax version
    version = data.FieldData["gprMax_version"].GetValue(0)
    # Number of voxels
    nl = data.FieldData["nx_ny_nz"]
    # Discretisation
    dl = data.FieldData["dx_dy_dz"]

    print("gprMax version:", data.FieldData["gprMax_version"].GetValue(0))
    print("nx_ny_nz:", nl)
    print("dx_dy_dz:", dl, end="\n\n")

    ################
    # Display data #
    ################

    # Get active view
    pv_view = GetActiveView()
    pv_view.AxesGrid.Visibility = 1  # Show Data Axes Grid

    # Get display properties
    pv_disp = GetDisplayProperties(pv_source, view=pv_view)

    # Set scalar colouring
    ColorBy(pv_disp, COLOUR_SCALARS)

    # Materials
    print("Loading materials... ", end="\t")
    if is_valid_key("material_ids", data.FieldData):
        threshold_materials(pv_source, pv_view, data.FieldData["material_ids"])
        print("Done.")
    else:
        print("No materials to load.")

    # Display any sources
    print("Loading sources... ", end="\t\t")
    if is_valid_key("source_ids", data.FieldData) and is_valid_key("sources", data.FieldData):
        create_box_sources(data.FieldData["source_ids"], data.FieldData["sources"], dl)
        print("Done.")
    else:
        print("No sources to load.")

    # Display any receivers
    print("Loading receivers... ", end="\t")
    if is_valid_key("receiver_ids", data.FieldData) and is_valid_key("receivers", data.FieldData):
        create_box_sources(data.FieldData["receiver_ids"], data.FieldData["receivers"], dl)
        print("Done.")
    else:
        print("No receivers to load.")

    # Display any PMLs
    print("Loading PMLs... ", end="\t\t")
    if is_valid_key("pml_thickness", data.FieldData):
        display_pmls(data.FieldData["pml_thickness"], dl, nl)
        print("Done.")
    else:
        print("No PMLs to load.")

    RenderAllViews()

    SetActiveSource(pv_source)
    Hide(pv_source)

    # Reset view to fit data
    pv_view.ResetCamera()

except HaltException as e:
    print(e)

print()
