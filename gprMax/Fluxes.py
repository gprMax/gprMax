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


class Flux(object):
    possible_normals = ['x', 'y', 'z']
    possible_direction = ['plus', 'minus']
    def __init__(self, G, normal, direction, bottom_left_corner, top_right_corner, wavelenghts):
        self.bottom_left_corner = bottom_left_corner
        self.top_right_corner = top_right_corner
        self.normal = normal
        self.direction = direction
        self.set_number_cells(G)
        self.wavelengths = wavelenghts
        self.omega = 2*np.pi * 299792458 / wavelenghts
        if G.scattering:
            if G.gpu is None:
                self.E_fft_transform_empty = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.E_fft_transform_scatt = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.H_fft_transform_empty = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
                self.H_fft_transform_scatt = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
            else:
                import pycuda.gpuarray as gpuarray
                self.E_fft_transform_empty = gpuarray.to_gpu(np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.E_fft_transform_scatt = gpuarray.to_gpu(np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.H_fft_transform_empty = gpuarray.to_gpu(np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                self.H_fft_transform_scatt = gpuarray.to_gpu(np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype))
                
        self.E_fft_transform = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)
        self.H_fft_transform = np.zeros((len(wavelenghts), self.cells_number[0], self.cells_number[1], self.cells_number[2], self.cells_number[3]), dtype= complextype)

        
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
                save_fields_fluxes_gpu(len(self.wavelengths),
                                        self.cells_number[0], self.cells_number[1], self.cells_number[2],
                                        G.Ex_gpu, G.Ey_gpu, G.Ez_gpu, G.Hx_gpu, G.Hy_gpu, G.Hz_gpu,
                                        self.wavelengths, self.E_fft_transform, self.H_fft_transform,
                                        iteration, G.dt
                )
        else:
            if G.gpu is None:
                if G.empty_sim:
                    save_fields_fluxes_pyx(
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, self.omega,
                        self.E_fft_transform_empty, self.H_fft_transform_empty,
                        self.cells_range['x'][0], self.cells_range['y'][0], self.cells_range['z'][0],
                        len(self.cells_range['x']), len(self.cells_range['y']), len(self.cells_range['z']), len(self.wavelengths),
                        int(self.bottom_left_corner[0]), int(self.bottom_left_corner[1]), int(self.bottom_left_corner[2]),
                        G.dt, G.nthreads, iteration
                    )
                else:
                    save_fields_fluxes_pyx(
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, self.omega,
                        self.E_fft_transform_scatt, self.H_fft_transform_scatt,
                        self.cells_range['x'][0], self.cells_range['y'][0], self.cells_range['z'][0],
                        len(self.cells_range['x']), len(self.cells_range['y']), len(self.cells_range['z']), len(self.wavelengths),
                        int(self.bottom_left_corner[0]), int(self.bottom_left_corner[1]), int(self.bottom_left_corner[2]),
                        G.dt, G.nthreads, iteration
                    )
            else:
                if G.empty_sim:
                    save_fields_fluxes_gpu(len(self.wavelengths),
                                        self.cells_number[0], self.cells_number[1], self.cells_number[2],
                                        G.Ex_gpu, G.Ey_gpu, G.Ez_gpu, G.Hx_gpu, G.Hy_gpu, G.Hz_gpu,
                                        self.wavelengths, self.E_fft_transform_empty, self.H_fft_transform_empty,
                                        iteration, G.dt
                    )
                else:
                    save_fields_fluxes_gpu(len(self.wavelengths),
                                        self.cells_number[0], self.cells_number[1], self.cells_number[2],
                                        G.Ex_gpu, G.Ey_gpu, G.Ez_gpu, G.Hx_gpu, G.Hy_gpu, G.Hz_gpu,
                                        self.wavelengths, self.E_fft_transform_scatt, self.H_fft_transform_scatt,
                                        iteration, G.dt
                    )

    def convert_to_scattering(self):
        self.E_fft_transform = self.E_fft_transform_scatt - self.E_fft_transform_empty
        self.H_fft_transform = self.H_fft_transform_scatt - self.H_fft_transform_empty


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
        self.Poynting_frequency_flux = np.real(np.array(self.Poynting_frequency_flux, dtype=floattype))
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
            grp['values'] = G.fluxes[i].Poynting_frequency_flux_incident
            grp['wavelengths'] = G.fluxes[i].wavelengths
            grp['normal'] = G.fluxes[i].normal
            grp['direction'] = G.fluxes[i].direction
            grp['x_cells'], grp['y_cells'], grp['z_cells'], dimension = G.fluxes[i].cells_number
    else:
        title = '/fluxes'
    for i in range(len(G.fluxes)):
        grp = f.create_group(title + '/fluxes/flux' + str(i + 1))
        grp['values'] = G.fluxes[i].Poynting_frequency_flux
        grp['wavelengths'] = G.fluxes[i].wavelengths
        grp['normal'] = G.fluxes[i].normal
        grp['direction'] = G.fluxes[i].direction
        grp['x_cells'], grp['y_cells'], grp['z_cells'], dimension = G.fluxes[i].cells_number
    grp = f.create_group(title + '/total_fluxes')
    grp['values'] = G.total_flux
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
    else:
        solve_gpu_fluxes(currentmodelrun, modelend, G)
    
    #Initializing everything once again and adding the new geometries
    G.initialise_geometry_arrays()        
    if G.gpu is None:
        G.initialise_field_arrays()
        for pml in G.pmls:
            pml.initialise_field_arrays()
    else:
        G.gpu_initialise_arrays
    process_geometrycmds(G.scatteringgeometry, G)
    pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
    build_pmls(G, pbar)
    pbar.close()
    build_electric_components(G.solid, G.rigidE, G.ID, G)
    build_magnetic_components(G.solid, G.rigidH, G.ID, G)
    G.initialise_std_update_coeff_arrays()
    if G.gpu is None:
        G.initialise_dispersive_arrays()
    else:
        G.gpu_initialise_dispersive_arrays
    process_materials(G)
    G.empty_sim = False

    #Run the simulation with scattering geometries
    if not G.gpu:
        tsolve = solve_cpu_fluxes(currentmodelrun, modelend, G)
    else:
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
            if G.cylindrical:
                raise CmdInputError("Magnetic dipole and transmission lines sources are not supported in cylindrical mode.")
            source.update_magnetic(iteration, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

        # Update electric field components
        # All materials are non-dispersive so do standard update
        if Material.maxpoles == 0:
            update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        elif Material.maxpoles == 1:
            assert not G.cylindrical, "Dispersive materials are not supported in cylindrical mode."
            update_electric_dispersive_1pole_A(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
        elif Material.maxpoles > 1:
            assert not G.cylindrical, "Dispersive materials are not supported in cylindrical mode."
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

    for flux in G.fluxes:
        flux.convert_to_scattering()
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
    
    #FFT functions
    kernel_fields_fluxes_gpu = SourceModule(kernel_template_store_outputs.substitute(REAL=cudafloattype, COMPLEX= cudacomplextype, NX_FIELDS=G.nx, NY_FIELDS=G.ny, NZ_FIELDS=G.nz, NC=3), options=compiler_opts)
    save_fields_fluxes_gpu = kernel_fields_fluxes_gpu.get_function("save_fields_fluxes_gpu")
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
            
        for flux in G.fluxes:
            flux.save_fields_fluxes(G, iteration, save_fields_fluxes_gpu)
    
    for flux in G.fluxes:
        flux.convert_to_scattering()

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