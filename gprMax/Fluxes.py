import numpy as np
from gprMax.grid import FDTDGrid
from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.utilities import round_value
import h5py
from gprMax._version import __version__
from gprMax.Fluxes_ext import save_fields_fluxes as save_fields_fluxes_pyx
from gprMax.exceptions import CmdInputError, GeneralError
from gprMax.utilities import timer
from gprMax.input_cmds_geometry import process_geometrycmds

from tqdm import tqdm
from colorama import init
from colorama import Fore
from colorama import Style
init()
from gprMax.fields_outputs import store_outputs
from gprMax.fields_updates_ext import update_electric
from gprMax.fields_updates_ext import update_magnetic
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_A
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_B
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_A
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_B
from gprMax.yee_cell_build_ext import build_magnetic_components, build_electric_components
from gprMax.materials import Material
from gprMax.materials import process_materials
from gprMax.utilities import get_terminal_width
import sys
import os

from gprMax.receivers import gpu_initialise_rx_arrays
from gprMax.receivers import gpu_get_rx_array
from gprMax.snapshots import Snapshot
from gprMax.snapshots import gpu_initialise_snapshot_array
from gprMax.snapshots import gpu_get_snapshot_array
from gprMax.snapshots_gpu import kernel_template_store_snapshot
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.source_updates_gpu import kernels_template_sources
from gprMax.fields_updates_gpu import kernels_template_fields
from gprMax.utilities import round32
from importlib import import_module
from gprMax.utilities import human_size
from gprMax.fields_outputs import kernel_template_store_outputs


from gprMax.pml import build_pmls
from gprMax.constants import cudafloattype
from gprMax.constants import cudacomplextype
from gprMax.Fluxes_ext_gpu import kernel_fields_fluxes_gpu

class Flux(object):
    possible_normals = ['x', 'y', 'z']
    possible_direction = ['plus', 'minus']
    def __init__(self, G: FDTDGrid, normal, direction, bottom_left_corner, top_right_corner, wavelengths):

        self.bottom_left_corner = bottom_left_corner
        self.top_right_corner = top_right_corner
        self.normal = normal
        self.direction = direction
        self.set_number_cells(G)
        self.wavelengths = wavelengths
        self.omega = 2*np.pi * 299792458 / wavelengths
        if G.scattering:
            if G.gpu is None:
                self.E_fft_transform_empty = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.E_fft_transform_scatt = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.H_fft_transform_empty = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.H_fft_transform_scatt = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
            else:
                pass # Initialized directly inside the solve_gpu_fluxes function     
        self.E_fft_transform = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
        self.H_fft_transform = np.zeros((len(wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
        self.tpb = G.tpb
        self.bpg = (int(np.ceil((self.cells_number[0] * self.cells_number[1] * self.cells_number[2]) / self.tpb[0])), 1, 1)


    def initialize_fft_arrays_gpu(self, G: FDTDGrid):
        import pycuda.gpuarray as gpuarray
        self.E_fft_transform_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
        self.H_fft_transform_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
        if G.scattering:
                self.E_fft_transform_empty_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.E_fft_transform_scatt_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.H_fft_transform_empty_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.H_fft_transform_scatt_gpu = gpuarray.to_gpu(np.zeros((len(self.wavelengths), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))

    def set_number_cells(self, G: FDTDGrid):
        """Defines the cells that are part of the surface flux."""

        Nx = -self.bottom_left_corner[0] + self.top_right_corner[0] +1
        Ny = -self.bottom_left_corner[1] + self.top_right_corner[1] +1
        Nz = -self.bottom_left_corner[2] + self.top_right_corner[2] +1

        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise AssertionError("Nx, Ny, and Nz must be positive")

        if self.normal == 'x':
            assert self.bottom_left_corner[0] == self.top_right_corner[0], "For normal 'x', the two x coordinates must be equal"
        elif self.normal == 'y':
            assert self.bottom_left_corner[1] == self.top_right_corner[1], "For normal 'y', the two y coordinates must be equal"
        elif self.normal == 'z':
            assert self.bottom_left_corner[2] == self.top_right_corner[2], "For normal 'z', the two z coordinates must be equal"

        self.cells_range = {'x': np.arange(0, -int(self.bottom_left_corner[0]) + int(self.top_right_corner[0])) if -int(self.bottom_left_corner[0]) + int(self.top_right_corner[0]) != 0 else [0],
                            'y': np.arange(0, -int(self.bottom_left_corner[1]) + int(self.top_right_corner[1])) if -int(self.bottom_left_corner[1]) + int(self.top_right_corner[1]) != 0 else [0],
                            'z': np.arange(0, -int(self.bottom_left_corner[2]) + int(self.top_right_corner[2])) if -int(self.bottom_left_corner[2]) + int(self.top_right_corner[2]) != 0 else [0]}

        self.cells_number = (int(Nx), int(Ny), int(Nz), 3)

    def save_fields_fluxes(self, G: FDTDGrid, iteration, save_fields_fluxes_gpu= None):   
        if not G.scattering:
            if G.gpu is None:
                save_fields_fluxes_pyx(
                    G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, self.omega,
                    self.E_fft_transform, self.H_fft_transform,
                    self.cells_range['x'][0], self.cells_range['y'][0], self.cells_range['z'][0],
                    len(self.cells_range['x']), len(self.cells_range['y']), len(self.cells_range['z']), len(self.wavelengths),
                    int(self.bottom_left_corner[0]), int(self.bottom_left_corner[1]), int(self.bottom_left_corner[2]),
                    G.dt, G.nthreads, iteration
                )
            else:
                save_fields_fluxes_gpu(
                        np.int32(len(self.wavelengths)), np.int32(self.cells_number[0]), np.int32(self.cells_number[1]), np.int32(self.cells_number[2]),
                        np.int32(self.bottom_left_corner[0]), np.int32(self.bottom_left_corner[1]), np.int32(self.bottom_left_corner[2]),
                        G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata, G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                        self.E_fft_transform_gpu.gpudata, self.H_fft_transform_gpu.gpudata,
                        np.int32(iteration), np.float32(G.dt),
                        block=self.tpb, grid=self.bpg
                )
                #After this, we have to convert those lists back to CPU lists
        else:
            if G.gpu is None:
                if G.empty_sim:
                    # print(G.Ez[50,50,50])
                    save_fields_fluxes_pyx(
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, self.omega,
                        self.E_fft_transform_empty, self.H_fft_transform_empty,
                        self.cells_range['x'][0], self.cells_range['y'][0], self.cells_range['z'][0],
                        len(self.cells_range['x']), len(self.cells_range['y']), len(self.cells_range['z']), len(self.wavelengths),
                        int(self.bottom_left_corner[0]), int(self.bottom_left_corner[1]), int(self.bottom_left_corner[2]),
                        G.dt, G.nthreads, iteration
                    )
                else:
                    # print(G.Ez[50,50,50])
                    save_fields_fluxes_pyx(
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, self.omega,
                        self.E_fft_transform_scatt, self.H_fft_transform_scatt,
                        self.cells_range['x'][0], self.cells_range['y'][0], self.cells_range['z'][0],
                        len(self.cells_range['x']), len(self.cells_range['y']), len(self.cells_range['z']), len(self.wavelengths),
                        int(self.bottom_left_corner[0]), int(self.bottom_left_corner[1]), int(self.bottom_left_corner[2]),
                        G.dt, G.nthreads, iteration
                    )
                #Because the empty simulation goes first, we substract the fields here
                self.E_fft_transform = self.E_fft_transform_scatt - self.E_fft_transform_empty
                self.H_fft_transform = self.H_fft_transform_scatt - self.H_fft_transform_empty

                file = open('Simulation_E_fft.txt', 'w')
                file.write(np.array2string(self.E_fft_transform))
                file.close()
                file = open('Simulation_H_fft.txt', 'w')
                file.write(np.array2string(self.H_fft_transform))
                file.close()
            else:
                if G.empty_sim:
                    # print(G.Ez_gpu.get()[50,50,50])
                    save_fields_fluxes_gpu(
                        np.int32(len(self.wavelengths)), np.int32(self.cells_number[0]), np.int32(self.cells_number[1]), np.int32(self.cells_number[2]),
                        np.int32(self.bottom_left_corner[0]), np.int32(self.bottom_left_corner[1]), np.int32(self.bottom_left_corner[2]),
                        G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata, G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                        self.E_fft_transform_empty_gpu.gpudata, self.H_fft_transform_empty_gpu.gpudata,
                        np.int32(iteration), np.float32(G.dt),
                        block=self.tpb, grid=self.bpg
                    )
                else:
                    # print(G.Ez_gpu.get()[50,50,50])
                    save_fields_fluxes_gpu(
                        np.int32(len(self.wavelengths)), np.int32(self.cells_number[0]), np.int32(self.cells_number[1]), np.int32(self.cells_number[2]),
                        np.int32(self.bottom_left_corner[0]), np.int32(self.bottom_left_corner[1]), np.int32(self.bottom_left_corner[2]),
                        G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata, G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                        self.E_fft_transform_scatt_gpu.gpudata, self.H_fft_transform_scatt_gpu.gpudata,
                        np.int32(iteration), np.float32(G.dt),
                        block=self.tpb, grid=self.bpg
                    )
                # Here, the conversion is done outside of the function because there would be a huge difference between doing this once or N_iteration * N_fluxes
                # as this is a CPU operation.

    def converting_back_to_cpu(self, G: FDTDGrid):
        if G.scattering:
            if not G.empty_sim:
                self.E_fft_transform = self.E_fft_transform_scatt_gpu.get() - self.E_fft_transform_empty
                self.H_fft_transform = self.H_fft_transform_scatt_gpu.get() - self.H_fft_transform_empty
                file = open('Simulation_E_fft.txt', 'w')
                file.write(np.array2string(self.E_fft_transform))
                file.close()

                file = open('Simulation_H_fft.txt', 'w')
                file.write(np.array2string(self.H_fft_transform))
                file.close()
            else:
                self.E_fft_transform_empty = self.E_fft_transform_empty_gpu.get() #For incident fluxes
                self.H_fft_transform_empty = self.H_fft_transform_empty_gpu.get() #For incident fluxes
        else:
            self.E_fft_transform = self.E_fft_transform_gpu.get()
            self.H_fft_transform = self.H_fft_transform_gpu.get()


    def calculate_Poynting_frequency_flux(self, G, incident= False):
        if not incident:
            #Then calculate the Poynting vector
            self.Poynting_frequency = np.cross(np.conjugate(self.E_fft_transform), self.H_fft_transform, axis=-1)
        else:
            self.Poynting_frequency = np.cross(np.conjugate(self.E_fft_transform_empty), self.H_fft_transform_empty, axis=-1)

        #Finally calculate the flux
        self.Poynting_frequency_flux = []
        for f in range(len(self.omega)):
            if self.normal == 'x':
                self.Poynting_frequency_flux.append(np.sum(self.Poynting_frequency[f, :, :, :, 0] * G.dy * G.dz, axis=None))
            elif self.normal == 'y':
                self.Poynting_frequency_flux.append(np.sum(self.Poynting_frequency[f, :, :, :, 1] * G.dx * G.dz, axis=None))
            elif self.normal == 'z':
                self.Poynting_frequency_flux.append(np.sum(self.Poynting_frequency[f, :, :, :, 2] * G.dx * G.dy, axis=None))
        self.Poynting_frequency_flux = np.real(np.array(self.Poynting_frequency_flux, dtype=complextype))
        if self.direction == 'minus':
            self.Poynting_frequency_flux *= -1
        if incident:
            self.Poynting_frequency_flux_incident = np.copy(self.Poynting_frequency_flux)
            self.Poynting_frequency_flux = None


def save_file_h5py(outputfile, G: FDTDGrid):
    f = h5py.File(outputfile, 'w')
    f.attrs['gprMax'] = __version__
    f.attrs['Title'] = G.title
    f.attrs['Iterations'] = G.iterations
    f.attrs['dt'] = G.dt
    f.attrs['n_surfaces'] = len(G.fluxes)
    wavelengths = G.fluxes[0].wavelengths

    if G.scattering:
        title = '/scattering'
        for i in range(len(G.fluxes)):
            grp = f.create_group(title + '/incidents/incident' + str(i + 1))
            cst = 1
            if G.fluxes[i].direction == 'minus':
                cst = -1
            grp['values'] = cst * G.fluxes[i].Poynting_frequency_flux_incident
            grp['wavelengths'] = G.fluxes[i].wavelengths
            grp['normal'] = G.fluxes[i].normal
            grp['direction'] = G.fluxes[i].direction
            grp['x_cells'], grp['y_cells'], grp['z_cells'], dimension = G.fluxes[i].cells_number
    if len(G.fluxes_single) != 0:
        title = '/fluxes'
        for i in range(len(G.fluxes_single)):
            grp = f.create_group(title + '/flux' + str(i + 1))
            grp['values'] = G.fluxes[i].Poynting_frequency_flux
            grp['wavelengths'] = G.fluxes[i].wavelengths
            grp['normal'] = G.fluxes[i].normal
            grp['direction'] = G.fluxes[i].direction
            grp['x_cells'], grp['y_cells'], grp['z_cells'], dimension = G.fluxes[i].cells_number
    if len(G.fluxes_box) != 0:
        title = '/boxes'
        for i in range(len(G.fluxes_box)):
            grp = f.create_group(title + '/box' + str(i + 1))
            grp['values'] = G.total_flux[i]
            grp['wavelengths'] = wavelengths
            grp = f.create_group('/constants')
            grp['dt'] = G.dt
            grp['dx'] = G.dx
            grp['dy'] = G.dy
            grp['dz'] = G.dz

    if G.scattering:
        grp = f.create_group('/scattering/incident')
        grp['values'] = G.fluxes[i].Poynting_frequency_flux_incident
        grp['wavelengths'] = G.fluxes[i].wavelengths
        grp['normal'] = G.fluxes[i].normal
        grp['direction'] = G.fluxes[i].direction
        grp['x_cells'], grp['y_cells'], grp['z_cells'], dimension = G.fluxes[i].cells_number


        
def solve_scattering(currentmodelrun, modelend, G:FDTDGrid):
    memsolve = None
    tstart = timer()    
    if len(G.scatteringgeometry) == 0:
        raise GeneralError("No geometry input for the scattering geometry !")
    box_settings = ''.join(G.box_fluxes_enumerate) if len(G.box_fluxes_enumerate) != 0 else 'None \n'
    geometry_settings = ''
    for key in G.scattering_geometrycmds:
        lis = G.scattering_geometrycmds[key]
        if len(lis) != 0:
            geometry_settings += '  - ' + str(key) + ': ' + ' '.join(lis) + ' \n'
    print(Fore.GREEN + "Scattering: \n  Scattering geometry: \n" + geometry_settings + "  Box settings: " + box_settings + Fore.RESET)

    #Run one simulation without the scattering geometry
    if G.gpu is None:
        solve_cpu_fluxes(currentmodelrun, modelend, G)
        #Initializing everything once again and adding the new geometries
        G.initialise_geometry_arrays()        
        
        #We need to re-build the pmls, so we have to empty G.pmls   
        for _ in range(len(G.pmls)):
            G.pmls.pop()

        pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
        build_pmls(G, pbar)
        pbar.close()

        G.initialise_field_arrays()
        print(Fore.BLUE +'\n==================Scattering geometries : ' +  ' '.join(G.scatteringgeometry) + '=================\n' + Fore.RESET)
        process_geometrycmds(G.scatteringgeometry, G)
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        G.initialise_std_update_coeff_arrays()
        G.initialise_dispersive_arrays()
        process_materials(G)
        G.empty_sim = False

        #Run the simulation with scattering geometries
        solve_cpu_fluxes(currentmodelrun, modelend, G)
    else:

        solve_gpu_fluxes(currentmodelrun, modelend, G)   

        #We need to re-build the pmls, so we have to empty G.pmls   
        for _ in range(len(G.pmls)):
            G.pmls.pop()

        process_geometrycmds(G.scatteringgeometry, G)

        pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
        build_pmls(G, pbar)
        pbar.close()
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        G.initialise_std_update_coeff_arrays()
        G.initialise_dispersive_arrays()
        process_materials(G)
        G.empty_sim = False

        #Run the simulation with scattering geometries
        tsolve, memsolve = solve_gpu_fluxes(currentmodelrun, modelend, G)

    tsolve = timer() - tstart
    return tsolve, memsolve

def solve_cpu_fluxes(currentmodelrun, modelend, G: FDTDGrid):
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
    tsolvestart = timer()
    message = "Running scattering simulation without scattering geometries" if G.empty_sim else "Running scattering simulation with scattering geometries"
    for iteration in tqdm(range(G.iterations), desc= message + ', model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars):
        # Store field component values for every receiver and transmission line
        store_outputs(iteration, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)
        # Store any snapshots
        if not G.empty_sim:
            for snap in G.snapshots:
                if snap.time == iteration + 1:
                    snap.store(G)

        # Update magnetic field components
        update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)


        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.update_magnetic(G) #No need to check for cylindrical mode here as there exists PML_cyl class with the same method
        
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
            pml.update_electric(G) #No need to check for cylindrical mode here as there exists PML_cyl class with the same method

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
        
        for flux in G.fluxes:
            flux.save_fields_fluxes(G, iteration)

    tsolve = timer() - tsolvestart
    return tsolve

def solve_gpu_fluxes(currentmodelrun, modelend, G: FDTDGrid):
    """Solving using FDTD method on GPU. Implemented using Nvidia CUDA.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving
        memsolve (int): memory usage on final iteration in bytes
    """

    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    drv.init()

    # Suppress nvcc warnings on Windows
    if sys.platform == 'win32':
        compiler_opts = ['-w']
    else:
        compiler_opts = None

    # Create device handle and context on specifc GPU device (and make it current context)
    dev = drv.Device(G.gpu.deviceID)
    ctx = dev.make_context()

    # Electric and magnetic field updates - prepare kernels, and get kernel functions
    if Material.maxpoles > 0:
        kernels_fields = SourceModule(kernels_template_fields.substitute(REAL=cudafloattype, COMPLEX=cudacomplextype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=G.updatecoeffsdispersive.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=G.Tx.shape[1], NY_T=G.Tx.shape[2], NZ_T=G.Tx.shape[3]), options=compiler_opts)
    else:   # Set to one any substitutions for dispersive materials
        kernels_fields = SourceModule(kernels_template_fields.substitute(REAL=cudafloattype, COMPLEX=cudacomplextype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=1, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=1, NY_T=1, NZ_T=1), options=compiler_opts)
    update_e_gpu = kernels_fields.get_function("update_e")
    update_h_gpu = kernels_fields.get_function("update_h")

    # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for fields kernels
    updatecoeffsE = kernels_fields.get_global('updatecoeffsE')[0]
    updatecoeffsH = kernels_fields.get_global('updatecoeffsH')[0]
    if G.updatecoeffsE.nbytes + G.updatecoeffsH.nbytes > G.gpu.constmem:
        raise GeneralError('Too many materials in the model to fit onto constant memory of size {} on {} - {} GPU'.format(human_size(G.gpu.constmem), G.gpu.deviceID, G.gpu.name))
    else:
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)

    # Electric and magnetic field updates - dispersive materials - get kernel functions and initialise array on GPU
    if Material.maxpoles > 0:  # If there are any dispersive materials (updates are split into two parts as they require present and updated electric field values).
        update_e_dispersive_A_gpu = kernels_fields.get_function("update_e_dispersive_A")
        update_e_dispersive_B_gpu = kernels_fields.get_function("update_e_dispersive_B")
        G.gpu_initialise_dispersive_arrays()

    # Electric and magnetic field updates - set blocks per grid and initialise field arrays on GPU
    G.gpu_set_blocks_per_grid()
    G.gpu_initialise_arrays()

    # PML updates
    if G.pmls:
        # Prepare kernels
        pmlmodulelectric = 'gprMax.pml_updates.pml_updates_electric_' + G.pmlformulation + '_gpu'
        kernelelectricfunc = getattr(import_module(pmlmodulelectric), 'kernels_template_pml_electric_' + G.pmlformulation)
        pmlmodulemagnetic = 'gprMax.pml_updates.pml_updates_magnetic_' + G.pmlformulation + '_gpu'
        kernelmagneticfunc = getattr(import_module(pmlmodulemagnetic), 'kernels_template_pml_magnetic_' + G.pmlformulation)
        kernels_pml_electric = SourceModule(kernelelectricfunc.substitute(REAL=cudafloattype, N_updatecoeffsE=G.updatecoeffsE.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        kernels_pml_magnetic = SourceModule(kernelmagneticfunc.substitute(REAL=cudafloattype, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsH.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for PML kernels
        updatecoeffsE = kernels_pml_electric.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernels_pml_magnetic.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        # Set block per grid, initialise arrays on GPU, and get kernel functions
        for pml in G.pmls:
            pml.gpu_initialise_arrays()
            pml.gpu_get_update_funcs(kernels_pml_electric, kernels_pml_magnetic)
            pml.gpu_set_blocks_per_grid(G)

    # Receivers
    if G.rxs:
        # Initialise arrays on GPU
        rxcoords_gpu, rxs_gpu = gpu_initialise_rx_arrays(G)
        # Prepare kernel and get kernel function
        kernel_store_outputs = SourceModule(kernel_template_store_outputs.substitute(REAL=cudafloattype, NY_RXCOORDS=3, NX_RXS=6, NY_RXS=G.iterations, NZ_RXS=len(G.rxs), NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1), options=compiler_opts)
        store_outputs_gpu = kernel_store_outputs.get_function("store_outputs")

    # Sources - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.voltagesources + G.hertziandipoles + G.magneticdipoles:
        kernels_sources = SourceModule(kernels_template_sources.substitute(REAL=cudafloattype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_SRCINFO=4, NY_SRCWAVES=G.iterations, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for source kernels
        updatecoeffsE = kernels_sources.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernels_sources.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        if G.hertziandipoles:
            srcinfo1_hertzian_gpu, srcinfo2_hertzian_gpu, srcwaves_hertzian_gpu = gpu_initialise_src_arrays(G.hertziandipoles, G)
            update_hertzian_dipole_gpu = kernels_sources.get_function("update_hertzian_dipole")
        if G.magneticdipoles:
            srcinfo1_magnetic_gpu, srcinfo2_magnetic_gpu, srcwaves_magnetic_gpu = gpu_initialise_src_arrays(G.magneticdipoles, G)
            update_magnetic_dipole_gpu = kernels_sources.get_function("update_magnetic_dipole")
        if G.voltagesources:
            srcinfo1_voltage_gpu, srcinfo2_voltage_gpu, srcwaves_voltage_gpu = gpu_initialise_src_arrays(G.voltagesources, G)
            update_voltage_source_gpu = kernels_sources.get_function("update_voltage_source")

    save_fields_fluxes_gpu = []
    for i in range(len(G.fluxes)):
        G.fluxes[i].initialize_fft_arrays_gpu(G)
        kernelfieldsfluxesgpu = SourceModule(
            kernel_fields_fluxes_gpu.substitute(REAL=cudafloattype, COMPLEX= cudacomplextype,
                                                x_begin= int(G.fluxes[i].bottom_left_corner[0]), y_begin= int(G.fluxes[i].bottom_left_corner[1]), z_begin= int(G.fluxes[i].bottom_left_corner[2]),
                                                NX_FLUX= G.fluxes[i].cells_number[0], NY_FLUX= G.fluxes[i].cells_number[1], NZ_FLUX= G.fluxes[i].cells_number[2],
                                                NF = len(G.fluxes[i].wavelengths), NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NC=3,
                                                OMEGA= 'omega_{}'.format(i), EXP_FACTOR= 'exp_factor_{}'.format(i), PHASE= 'phase_{}'.format(i),
                                                SIN_OMEGA= 'sin_omega_{}'.format(i), COS_OMEGA= 'cos_omega_{}'.format(i), NORM= 'norm_{}'.format(i),
                                                IDX_5DX= 'idx_5d_{}'.format(i), IDX_5DY= 'idy_5d_{}'.format(i), IDX_5DZ= 'idz_5d_{}'.format(i), IDX_3D= 'idx_3d_{}'.format(i)
                                                ),
                                                options=compiler_opts)
        save_fields_fluxes_gpu.append(kernelfieldsfluxesgpu.get_function("save_fields_flux"))
        omega = kernelfieldsfluxesgpu.get_global('omega_{}'.format(i))[0]
        drv.memcpy_htod(omega, G.fluxes[i].omega)

    # Snapshots - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.snapshots:
        # Initialise arrays on GPU
        snapEx_gpu, snapEy_gpu, snapEz_gpu, snapHx_gpu, snapHy_gpu, snapHz_gpu = gpu_initialise_snapshot_array(G)
        # Prepare kernel and get kernel function
        kernel_store_snapshot = SourceModule(kernel_template_store_snapshot.substitute(REAL=cudafloattype, NX_SNAPS=Snapshot.nx_max, NY_SNAPS=Snapshot.ny_max, NZ_SNAPS=Snapshot.nz_max, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1), options=compiler_opts)
        store_snapshot_gpu = kernel_store_snapshot.get_function("store_snapshot")

    # Iteration loop timer
    iterstart = drv.Event()
    iterend = drv.Event()
    iterstart.record()

    for iteration in tqdm(range(G.iterations), desc='Running simulation, model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars):

        # Get GPU memory usage on final iteration
        if iteration == G.iterations - 1:
            memsolve = drv.mem_get_info()[1] - drv.mem_get_info()[0]

        # Store field component values for every receiver
        if G.rxs:
            store_outputs_gpu(np.int32(len(G.rxs)), np.int32(iteration),
                              rxcoords_gpu.gpudata, rxs_gpu.gpudata,
                              G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                              G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                              block=(1, 1, 1), grid=(round32(len(G.rxs)), 1, 1))

        # Store any snapshots
        for i, snap in enumerate(G.snapshots):
            if snap.time == iteration + 1:
                if not G.snapsgpu2cpu:
                    store_snapshot_gpu(np.int32(i), np.int32(snap.xs),
                                       np.int32(snap.xf), np.int32(snap.ys),
                                       np.int32(snap.yf), np.int32(snap.zs),
                                       np.int32(snap.zf), np.int32(snap.dx),
                                       np.int32(snap.dy), np.int32(snap.dz),
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       snapEx_gpu.gpudata, snapEy_gpu.gpudata, snapEz_gpu.gpudata,
                                       snapHx_gpu.gpudata, snapHy_gpu.gpudata, snapHz_gpu.gpudata,
                                       block=Snapshot.tpb, grid=Snapshot.bpg)
                else:
                    store_snapshot_gpu(np.int32(0), np.int32(snap.xs),
                                       np.int32(snap.xf), np.int32(snap.ys),
                                       np.int32(snap.yf), np.int32(snap.zs),
                                       np.int32(snap.zf), np.int32(snap.dx),
                                       np.int32(snap.dy), np.int32(snap.dz),
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       snapEx_gpu.gpudata, snapEy_gpu.gpudata, snapEz_gpu.gpudata,
                                       snapHx_gpu.gpudata, snapHy_gpu.gpudata, snapHz_gpu.gpudata,
                                       block=Snapshot.tpb, grid=Snapshot.bpg)
                    gpu_get_snapshot_array(snapEx_gpu.get(), snapEy_gpu.get(), snapEz_gpu.get(),
                                           snapHx_gpu.get(), snapHy_gpu.get(), snapHz_gpu.get(), 0, snap)

        # Update magnetic field components
        update_h_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                     G.ID_gpu.gpudata, G.Hx_gpu.gpudata, G.Hy_gpu.gpudata,
                     G.Hz_gpu.gpudata, G.Ex_gpu.gpudata, G.Ey_gpu.gpudata,
                     G.Ez_gpu.gpudata, block=G.tpb, grid=G.bpg)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_magnetic(G)

        # Update magnetic field components for magetic dipole sources
        if G.magneticdipoles:
            update_magnetic_dipole_gpu(np.int32(len(G.magneticdipoles)), np.int32(iteration),
                                       floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                       srcinfo1_magnetic_gpu.gpudata, srcinfo2_magnetic_gpu.gpudata,
                                       srcwaves_magnetic_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.magneticdipoles)), 1, 1))

        # Update electric field components
        # If all materials are non-dispersive do standard update
        if Material.maxpoles == 0:
            update_e_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz), G.ID_gpu.gpudata,
                         G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                         G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                         block=G.tpb, grid=G.bpg)
        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            update_e_dispersive_A_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(Material.maxpoles), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                      block=G.tpb, grid=G.bpg)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_electric(G)

        # Update electric field components for voltage sources
        if G.voltagesources:
            update_voltage_source_gpu(np.int32(len(G.voltagesources)), np.int32(iteration),
                                      floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                      srcinfo1_voltage_gpu.gpudata, srcinfo2_voltage_gpu.gpudata,
                                      srcwaves_voltage_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=(1, 1, 1), grid=(round32(len(G.voltagesources)), 1, 1))

        # Update electric field components for Hertzian dipole sources (update any Hertzian dipole sources last)
        if G.hertziandipoles:
            update_hertzian_dipole_gpu(np.int32(len(G.hertziandipoles)), np.int32(iteration),
                                       floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                       srcinfo1_hertzian_gpu.gpudata, srcinfo2_hertzian_gpu.gpudata,
                                       srcwaves_hertzian_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.hertziandipoles)), 1, 1))

        # If there are any dispersive materials do 2nd part of dispersive update (it is split into two parts as it requires present and updated electric field values). Therefore it can only be completely updated after the electric field has been updated by the PML and source updates.
        if Material.maxpoles > 0:
            update_e_dispersive_B_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(Material.maxpoles), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=G.tpb, grid=G.bpg)
            
        for i in range(len(G.fluxes)):
            G.fluxes[i].save_fields_fluxes(G, iteration, save_fields_fluxes_gpu[i])

    for flux in G.fluxes:
        flux.converting_back_to_cpu(G)
    # Copy output from receivers array back to correct receiver objects
    if G.rxs:
        gpu_get_rx_array(rxs_gpu.get(), rxcoords_gpu.get(), G)

    # Copy data from any snapshots back to correct snapshot objects
    if G.snapshots and not G.snapsgpu2cpu:
        for i, snap in enumerate(G.snapshots):
            gpu_get_snapshot_array(snapEx_gpu.get(), snapEy_gpu.get(), snapEz_gpu.get(),
                                   snapHx_gpu.get(), snapHy_gpu.get(), snapHz_gpu.get(), i, snap)

    iterend.record()
    iterend.synchronize()
    tsolve = iterstart.time_till(iterend) * 1e-3

    # Remove context from top of stack and delete
    ctx.pop()
    del ctx

    return tsolve, memsolve