import datetime
from importlib import import_module
import itertools
import os
import psutil
import sys

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm

from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.constants import cudafloattype
from gprMax.constants import cudacomplextype
from gprMax.exceptions import GeneralError

from gprMax.fields_outputs import store_outputs
from gprMax.fields_outputs import kernel_template_store_outputs
from gprMax.fields_outputs import write_hdf5_outputfile

from gprMax.fields_updates_ext import update_electric
from gprMax.fields_updates_ext import update_magnetic
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_A
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_B
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_A
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_B
from gprMax.fields_updates_gpu import kernels_template_fields

from gprMax.grid import FDTDGrid
from gprMax.grid import dispersion_analysis

from gprMax.input_cmds_geometry import process_geometrycmds
from gprMax.input_cmds_file import process_python_include_code
from gprMax.input_cmds_file import write_processed_file
from gprMax.input_cmds_file import check_cmd_names
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.input_cmds_singleuse import process_singlecmds
from gprMax.materials import Material
from gprMax.materials import process_materials
from gprMax.pml import CFS
from gprMax.pml import PML
from gprMax.pml import build_pmls
from gprMax.receivers import gpu_initialise_rx_arrays
from gprMax.receivers import gpu_get_rx_array
from gprMax.snapshots import Snapshot
from gprMax.snapshots import gpu_initialise_snapshot_array
from gprMax.snapshots import gpu_get_snapshot_array
from gprMax.snapshots_gpu import kernel_template_store_snapshot
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.source_updates_gpu import kernels_template_sources
from gprMax.utilities import get_host_info
from gprMax.utilities import get_terminal_width
from gprMax.utilities import human_size
from gprMax.utilities import open_path_file
from gprMax.utilities import round32
from gprMax.utilities import timer
from gprMax.utilities import round_value
from gprMax.yee_cell_build_ext import build_electric_components
from gprMax.yee_cell_build_ext import build_magnetic_components
from gprMax.exceptions import CmdInputError

def get_cube_vertices(cx, cy, cz, G, padding=10):
    """
    Computes the 8 vertices of a cube centered at `center` with `padding`,
    clipped to be within the FDTD grid dimensions.

    Args:
        center (tuple): (cx, cy, cz) center of the cube in index units.
        padding (int): Half-length of the cube in grid cells.
        G (object): FDTDGrid object with attributes G.nx, G.ny, G.nz

    Returns:
        List of 8 vertex coordinate tuples.
    """
    # Clamp bounds within grid
    x0 = max(cx - padding, 0)
    x1 = min(cx + padding, G.nx - 1)
    y0 = max(cy - padding, 0)
    y1 = min(cy + padding, G.ny - 1)
    z0 = max(cz - padding, 0)
    z1 = min(cz + padding, G.nz - 1)

    # 8 vertices of the cuboid
    vertices = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return vertices

def solve_cpu_nfft(currentmodelrun, modelend, G, center = None, padding=10):
    """
    Solving using FDTD method on CPU. Parallelised using Cython (OpenMP) for
    electric and magnetic field updates, and PML updates.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving
    """
    if center is not None and len(center) != 3:
        raise CmdInputError('The center requires exactly three coordinates')
    cx = round_value(G.nx / 2) if center is None else round_value(center[0] / G.dx)
    cy = round_value(G.ny / 2) if center is None else round_value(center[1] / G.dy)
    cz = round_value(G.nz / 2) if center is None else round_value(center[2] / G.dz)

    vertices = get_cube_vertices(cx, cy, cz, G, padding = 30)
    print(G.nx, G.ny, G.nz)
    print(G.dx, G.dy, G.dz)
    print(vertices)
    f1 = Snapshot(vertices[0][0], vertices[0][1], vertices[0][2], vertices[3][0]+1, vertices[3][1], vertices[3][2], 1, 1, 1)
    f2 = Snapshot(vertices[0][0], vertices[0][1], vertices[0][2], vertices[6][0], vertices[6][1], vertices[6][2]+1, 1, 1, 1)
    f3 = Snapshot(vertices[0][0], vertices[0][1], vertices[0][2], vertices[5][0], vertices[5][1]+1, vertices[5][2], 1, 1, 1)
    f4 = Snapshot(vertices[1][0], vertices[1][1], vertices[1][2], vertices[7][0], vertices[7][1], vertices[7][2]+1, 1, 1, 1)
    f5 = Snapshot(vertices[2][0], vertices[2][1], vertices[2][2], vertices[7][0], vertices[7][1]+1, vertices[7][2], 1, 1, 1)
    f6 = Snapshot(vertices[4][0], vertices[4][1], vertices[4][2], vertices[7][0]+1, vertices[7][1], vertices[7][2], 1, 1, 1)
    f_list = [f1, f2, f3, f4, f5, f6]
    print(f1.sx, f1.sy, f1.sz)
    elec_list = [[] for _ in range(6)]
    mag_list = [[] for _ in range(6)]
    
    tsolvestart = timer()

    for iteration in tqdm(range(G.iterations), desc='Running simulation, model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars):
        # Store field component values for every receiver and transmission line
        store_outputs(iteration, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

        # Store any snapshots
        for snap in G.snapshots:
            if snap.time == iteration + 1:
                snap.store(G)

        # Update magnetic field components
        update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.update_magnetic(G)

        # Update magnetic field components from sources
        for source in G.transmissionlines + G.magneticdipoles:
            source.update_magnetic(iteration, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

        # Update electric field components
        # All materials are non-dispersive so do standard update
        if Material.maxpoles == 0:
            update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        elif Material.maxpoles == 1:
            update_electric_dispersive_1pole_A(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_A(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.update_electric(G)

        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in G.voltagesources + G.transmissionlines + G.hertziandipoles:
            source.update_electric(iteration, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)

        # If there are any dispersive materials do 2nd part of dispersive update
        # (it is split into two parts as it requires present and updated electric
        # field values). Therefore it can only be completely updated after the
        # electric field has been updated by the PML and source updates.
        if Material.maxpoles == 1:
            update_electric_dispersive_1pole_B(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_B(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)

        for f, e, m in zip(f_list, elec_list, mag_list):
            f.store_surface(G)
            e.append(f.electric)
            # print(f.electric.shape)
            m.append(f.magnetic)
            # print(f.magnetic.shape)

    np.savez_compressed(
        f'nfft_snapshots_model{currentmodelrun}.npz',
        electric=np.array(elec_list, dtype=object),
        magnetic=np.array(mag_list, dtype=object)
    )


    tsolve = timer() - tsolvestart

    return tsolve