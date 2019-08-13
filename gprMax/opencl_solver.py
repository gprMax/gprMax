import numpy as np 
import os 
import sys 
from tqdm import tqdm
import warnings
import time 

import jinja2
from string import Template
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

from gprMax.materials import Material, process_materials
from gprMax.receivers import gpu_initialise_rx_arrays, gpu_get_rx_array
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.utilities import get_terminal_width, round32, timer
from gprMax.snapshots import Snapshot, cl_initialise_snapshot_array, gpu_get_snapshot_array
from gprMax.opencl_el_kernels import pml_updates_electric_HORIPML as el_electric
from gprMax.opencl_el_kernels import pml_updates_magnetic_HORIPML as el_magnetic

class OpenClSolver(object):
    def __init__(self, platformIdx=None, deviceIdx=None, G=None, context=None, queue=None):
        self.context = context
        self.queue = queue
        self.G = G
        self.jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))

    def getDeviceParameters(self):
        warnings.warn("All sizes are in Bytes")
        deviceParam = {}
        deviceParam['GLOBAL_MEM_SIZE'] = self.devices[self.deviceIdx].global_mem_size
        deviceParam['LOCAL_MEM_SIZE'] = self.devices[self.deviceIdx].local_mem_size  
        deviceParam['MAX_COMPUTE_UNITS'] = self.devices[self.deviceIdx].max_compute_units
        deviceParam['MAX_CONSTANT_BUFFER_SIZE'] = self.devices[self.deviceIdx].max_constant_buffer_size 
        deviceParam['MAX_WORK_GROUP_SIZE'] = self.devices[self.deviceIdx].max_work_group_size  
        deviceParam['MAX_WORK_ITEM_DIMENSIONS'] = self.devices[self.deviceIdx].max_work_item_dimensions
        deviceParam['MAX_WORK_ITEM_SIZES'] = self.devices[self.deviceIdx].max_work_item_sizes 
        deviceParam['OPENCL_C_VERSION'] = self.devices[self.deviceIdx].opencl_c_version
        self.deviceParam = deviceParam

    def getPlatformNDevices(self, mpi_no_spawn, platformIdx=None, deviceIdx=None, gpu=False):
        # get the opencl supported platforms
        self.platforms = cl.get_platforms() 
        print("Following platform supporting OpenCl were discovered")
        for idx, platf in enumerate(self.platforms):
            print("{} Platform: {}".format(str(idx), str(platf.name)))
        
        # get the platform index
        if platformIdx is None:
            if mpi_no_spawn:
                self.platformIdx = 0
            else: 
                self.platformIdx = int(input("Platform to be chosen ?"))
            assert self.platformIdx in [i for i in range(len(self.platforms))]
        else:
            self.platformIdx = platformIdx

        # if the platform is NVIDIA specific, the overhead memory size is about 42 MB (memory profiling shows the fact)
        if "nvidia" in str(self.platforms[self.platformIdx].name).lower():
            self.nvidiaGPU = True 
        else:
            self.nvidiaGPU = False

        # get the devices
        print("Following devices supporting OpenCl were discovered for platform: {} with GPU Tag {}".format(str(self.platforms[self.platformIdx].name), gpu))
        if gpu:
            self.devices = self.platforms[self.platformIdx].get_devices(device_type=cl.device_type.GPU)
        else:
            self.devices = self.platforms[self.platformIdx].get_devices()
        
        if self.devices is None:
            print("No Devices Found")
            # abort
            return False

        for idx, dev in enumerate(self.devices):
            print("{} Devices: {}".format(str(idx), str(dev.name)))
        
        if len(self.devices) is 1:
            deviceIdx = 0

        # set the device index
        if deviceIdx is None:
            if mpi_no_spawn:
                self.deviceIdx = 0
            else: 
                self.deviceIdx = int(input("Device to be chosen?"))
            assert self.deviceIdx in [i for i in range(len(self.devices))]
        else:
            self.deviceIdx = deviceIdx

        print("Chosen Platform and Device")
        print("Platform: {}".format(str(self.platforms[self.platformIdx].name)))
        print("Devices: {}".format(str(self.devices[self.deviceIdx].name)))
        self.getDeviceParameters()

        return True

    def createContext(self):
        if self.context is None:
            print("Creating context...")
            self.context = cl.Context(devices=[self.devices[self.deviceIdx]])

        if self.queue is None:
            print("Creating the command queue...")
            try:
                self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            except:
                self.queue = cl.CommandQueue(self.context)
                print("Profiling not enabled")

        return
    
    def setDataTypes(self):
        self.datatypes = {
            'REAL': 'float',
            'COMPLEX': 'cfloat'
        }
        return

    def elwise_kernel_build(self):
        
        elwise_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_el_kernels'))
        
        if Material.maxpoles > 0:
            ny_matdispcoeffs = self.G.updatecoeffsdispersive.shape[1]
            nx_t = self.G.Tx.shape[1]
            ny_t = self.G.Tx.shape[2] 
            nz_t = self.G.Tx.shape[3]
        else:
            ny_matdispcoeffs = 1
            nx_t = 1
            ny_t = 1
            nz_t = 1

        # get the preamble
        common_kernel = elwise_jinja_env.get_template('common.cl').render(
            REAL = self.datatypes['REAL'],
            COMPLEX = self.datatypes['COMPLEX'],
            N_updatecoeffsE = self.G.updatecoeffsE.size,
            N_updatecoeffsH = self.G.updatecoeffsH.size,
            updateEVal = self.updateEVal,
            updateHVal = self.updateHVal,
            NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS = ny_matdispcoeffs,
            NX_FIELDS = self.G.Ex.shape[0],
            NY_FIELDS = self.G.Ex.shape[1],
            NZ_FIELDS = self.G.Ex.shape[2],
            NX_ID = self.G.ID.shape[1],
            NY_ID = self.G.ID.shape[2],
            NZ_ID = self.G.ID.shape[3],
            NX_T = nx_t,
            NY_T = ny_t,
            NZ_T = nz_t,
            NY_RXCOORDS=3,
            NX_RXS=6, 
            NY_RXS=self.G.iterations, 
            NZ_RXS=len(self.G.rxs),
            NY_SRCINFO=4, 
            NY_SRCWAVES=self.G.iterations
        )

        if Material.maxpoles > 0:
            update_e_dispersive_A_context = elwise_jinja_env.get_template('update_e_dispersive_A.cl').render(
                REAL = self.datatypes['REAL'],
                NX_FIELDS = self.G.Ex.shape[0],
                NY_FIELDS = self.G.Ex.shape[1],
                NZ_FIELDS = self.G.Ex.shape[2],
                NX_ID = self.G.ID.shape[1],
                NY_ID = self.G.ID.shape[2],
                NZ_ID = self.G.ID.shape[3],
                NX_T = nx_t,
                NY_T = ny_t,
                NZ_T = nz_t,                
            )
            self.update_e_dispersive_A = ElementwiseKernel(
                self.context,
                Template("int NX, int NY, int NZ, int MAXPOLES, __global const ${COMPLEX}_t* restrict updatecoeffsdispersive, __global ${COMPLEX}_t *Tx, __global ${COMPLEX}_t *Ty, __global ${COMPLEX}_t *Tz, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz").substitute({'REAL':self.datatypes['REAL'],'COMPLEX':self.datatypes['COMPLEX']}),
                update_e_dispersive_A_context,
                "update_e_dispersive_A",
                preamble=common_kernel
            )

            update_e_dispersive_B_context = elwise_jinja_env.get_template('update_e_dispersive_B.cl').render(
                REAL = self.datatypes['REAL'],
                NX_FIELDS = self.G.Ex.shape[0],
                NY_FIELDS = self.G.Ex.shape[1],
                NZ_FIELDS = self.G.Ex.shape[2],
                NX_ID = self.G.ID.shape[1],
                NY_ID = self.G.ID.shape[2],
                NZ_ID = self.G.ID.shape[3],
                NX_T = nx_t,
                NY_T = ny_t,
                NZ_T = nz_t,
            )
            self.update_e_dispersive_B = ElementwiseKernel(
                self.context,
                Template("int NX, int NY, int NZ, int MAXPOLES, __global const ${COMPLEX}_t* restrict updatecoeffsdispersive, __global ${COMPLEX}_t *Tx, __global ${COMPLEX}_t *Ty, __global ${COMPLEX}_t *Tz, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez").substitute({'REAL':self.datatypes['REAL'],'COMPLEX':self.datatypes['COMPLEX']}),
                update_e_dispersive_B_context,
                "update_e_dispersive_B",
                preamble=common_kernel
            )
            
        e_update_context = elwise_jinja_env.get_template('update_field_e.cl').render(
            NX_FIELDS = self.G.Ex.shape[0],
            NY_FIELDS = self.G.Ex.shape[1],
            NZ_FIELDS = self.G.Ex.shape[2],
            NX_ID = self.G.ID.shape[1],
            NY_ID = self.G.ID.shape[2],
            NZ_ID = self.G.ID.shape[3]            
        )
        self.update_e_field = ElementwiseKernel(
            self.context,
            Template("int NX, int NY, int NZ, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL * restrict Hx, __global const $REAL * restrict Hy, __global const $REAL * restrict Hz").substitute({'REAL':self.datatypes['REAL']}),
            e_update_context,
            "update_e_field",
            preamble=common_kernel
        )

        h_update_context = elwise_jinja_env.get_template('update_field_h.cl').render(
            NX_FIELDS = self.G.Ex.shape[0],
            NY_FIELDS = self.G.Ex.shape[1],
            NZ_FIELDS = self.G.Ex.shape[2],
            NX_ID = self.G.ID.shape[1],
            NY_ID = self.G.ID.shape[2],
            NZ_ID = self.G.ID.shape[3]               
        )
        self.update_h_field = ElementwiseKernel(
            self.context,
            Template("int NX, int NY, int NZ, __global const unsigned int* restrict ID, __global $REAL *Hx, __global $REAL *Hy, __global $REAL *Hz, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez").substitute({'REAL':self.datatypes['REAL']}),
            h_update_context,
            "update_h_field",
            preamble=common_kernel
        )

        # initialize the cl arrays
        self.G.cl_initialize_arrays(self.queue)

        # check for dispersive materials
        if Material.maxpoles > 0:
            self.G.cl_initialize_dispersive_arrays(self.queue)

        # keeps an account for estimate of memory that shall be used
        # only the PyOpenCl memory buffers which are transferred to devices are calculated
        self.memUsage = self.G.clMemoryUsage

        # if pmls 
        if self.G.pmls:

            # get the electric and magnetic kernel files
            self.pml_magnetic_update = {}
            self.pml_electric_update = {}

            for pml in self.G.pmls:
                pml.cl_set_workgroups(self.G)
                pml.cl_initialize_arrays(self.queue)
                self.memUsage += pml.clMemoryUsage
                function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                electric_context_dict = {
                    'order1_xminus' : el_electric.order1_xminus,
                    'order2_xminus' : el_electric.order2_xminus,
                    'order1_xplus' : el_electric.order1_xplus,
                    'order2_xplus' : el_electric.order2_xplus,
                    'order1_yminus' : el_electric.order1_yminus,
                    'order2_yminus' : el_electric.order2_yminus,
                    'order1_yplus' : el_electric.order1_yplus,
                    'order2_yplus' : el_electric.order2_yplus,
                    'order1_zminus' : el_electric.order1_zminus,
                    'order2_zminus' : el_electric.order2_zminus,
                    'order1_zplus' : el_electric.order1_zplus,
                    'order2_zplus' : el_electric.order2_zplus
                }

                electric_function_arg_dict = {
                    'order1_xminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d", 
                    'order2_xminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order1_xplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_xplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order1_yminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_yminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order1_yplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_yplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order1_zminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_zminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order1_zplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_zplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d"
                }

                self.pml_electric_update[function_name] = ElementwiseKernel(
                    self.context,
                    Template(electric_function_arg_dict[function_name]).substitute({'REAL':self.datatypes['REAL']}),
                    electric_context_dict[function_name].substitute({'REAL':self.datatypes['REAL']}),
                    "pml_updates_electric_HORIPML_{}".format(function_name),
                    preamble=common_kernel
                )

                magnetic_context_dict = {
                    'order1_xminus' : el_magnetic.order1_xminus,
                    'order2_xminus' : el_magnetic.order2_xminus,
                    'order1_xplus' : el_magnetic.order1_xplus,
                    'order2_xplus' : el_magnetic.order2_xplus,
                    'order1_yminus' : el_magnetic.order1_yminus,
                    'order2_yminus' : el_magnetic.order2_yminus,
                    'order1_yplus' : el_magnetic.order1_yplus,
                    'order2_yplus' : el_magnetic.order2_yplus,
                    'order1_zminus' : el_magnetic.order1_zminus,
                    'order2_zminus' : el_magnetic.order2_zminus,
                    'order1_zplus' : el_magnetic.order1_zplus,
                    'order2_zplus' : el_magnetic.order2_zplus
                }

                magnetic_function_arg_dict = {
                    "order1_xminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global $REAL *Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_xminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global $REAL *Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order1_xplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global $REAL *Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_xplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global $REAL *Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order1_yminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_yminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order1_yplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_yplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order1_zminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global $REAL *Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_zminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global $REAL *Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order1_zplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global $REAL *Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_zplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global $REAL *Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d"
                }

                self.pml_magnetic_update[function_name] = ElementwiseKernel(
                    self.context,
                    Template(magnetic_function_arg_dict[function_name]).substitute({'REAL':self.datatypes['REAL']}),
                    magnetic_context_dict[function_name].substitute({'REAL':self.datatypes['REAL']}),
                    "pml_updates_magnetic_HORIPML_{}".format(function_name),
                    preamble=common_kernel
                )

        if self.G.rxs:
            self.rxcoords_cl, self.rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
            self.memUsage += self.rxcoords_cl.nbytes + self.rxs_cl.nbytes
            store_field_context = elwise_jinja_env.get_template('store_outputs.cl').render()
            self.store_field = ElementwiseKernel(
                self.context,
                Template("int NRX, int iteration, __global const int* restrict rxcoords, __global $REAL *rxs, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz").substitute({'REAL':self.datatypes['REAL']}),
                store_field_context,
                "store_field",
                preamble=common_kernel
            )

        if self.G.voltagesources + self.G.hertziandipoles + self.G.magneticdipoles:
            if self.G.hertziandipoles:
                hertziandipoles_context = elwise_jinja_env.get_template('update_hertziandipole.cl').render(REAL=self.datatypes['REAL'])
                self.hertziandipoles_update = ElementwiseKernel(
                    self.context,
                    Template("int NHERTZDIPOLE, int iteration, $REAL dx, $REAL dy, $REAL dz, __global const int* restrict srcinfo1, __global const $REAL* restrict srcinfo2, __global const $REAL* restrict srcwaveforms, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global $REAL *Ez").substitute({'REAL':self.datatypes['REAL']}),
                    hertziandipoles_context,
                    "hertziandipoles_update",
                    preamble=common_kernel
                )
                self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_hertzian_cl.nbytes + self.srcinfo2_hertzian_cl.nbytes + self.srcwaves_hertzian_cl.nbytes

            if self.G.voltagesources:
                voltagesource_context = elwise_jinja_env.get_template('update_voltagesource.cl').render(REAL=self.datatypes['REAL'])
                self.voltagesource_update = ElementwiseKernel(
                    self.context,
                    Template("int NVOLTSRC, int iteration, $REAL dx, $REAL dy, $REAL dz, __global const int* restrict srcinfo1, __global const $REAL* restrict srcinfo2, __global const $REAL* restrict srcwaveforms, __global const unsigned int* restrict ID, __global $REAL *Ex, __global $REAL *Ey, __global $REAL *Ez").substitute({'REAL':self.datatypes['REAL']}),
                    voltagesource_context,
                    "voltagesource_update",
                    preamble=common_kernel
                )
                self.srcinfo1_voltage_cl, self.srcinfo2_voltage_cl, self.srcwaves_voltage_cl = gpu_initialise_src_arrays(self.G.voltagesources, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_voltage_cl.nbytes + self.srcinfo2_voltage_cl.nbytes + self.srcwaves_voltage_cl.nbytes

            if self.G.magneticdipoles:
                magneticdipole_context = elwise_jinja_env.get_template('update_magneticdipole.cl').render(REAL=self.datatypes['REAL'])
                self.magneticdipole_update = ElementwiseKernel(
                    self.context,
                    Template("int NMAGDIPOLE, int iteration, $REAL dx, $REAL dy, $REAL dz, __global const int* restrict srcinfo1, __global const $REAL* restrict srcinfo2, __global const $REAL* restrict srcwaveforms, __global const unsigned int* restrict ID, __global $REAL *Hx, __global $REAL *Hy, __global $REAL *Hz").substitute({'REAL':self.datatypes['REAL']}),
                    magneticdipole_context,
                    "magneticdipole_update",
                    preamble=common_kernel
                )
                self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl, self.srcwaves_magnetic_cl = gpu_initialise_src_arrays(self.G.magneticdipoles, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_magnetic_cl.nbytes + self.srcinfo2_magnetic_cl.nbytes + self.srcwaves_magnetic_cl.nbytes

        if self.G.snapshots:
            raise NotImplementedError


    def traditional_kernel_build(self):
        # set the jinja engine
        trad_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))
        
        # for field update
        if Material.maxpoles > 0:
            ny_matdispcoeffs = self.G.updatecoeffsdispersive.shape[1]
            nx_t = self.G.Tx.shape[1]
            ny_t = self.G.Tx.shape[2] 
            nz_t = self.G.Tx.shape[3]
        else:
            ny_matdispcoeffs = 1
            nx_t = 1
            ny_t = 1 
            nz_t = 1

        kernel_fields_text = trad_jinja_env.get_template('update_fields.cl').render(
            REAL = self.datatypes['REAL'],
            COMPLEX = self.datatypes['COMPLEX'],
            N_updatecoeffsE = self.G.updatecoeffsE.size,
            N_updatecoeffsH = self.G.updatecoeffsH.size,
            updateEVal = self.updateEVal,
            updateHVal = self.updateHVal,
            NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS = ny_matdispcoeffs,
            NX_FIELDS = self.G.Ex.shape[0],
            NY_FIELDS = self.G.Ex.shape[1],
            NZ_FIELDS = self.G.Ex.shape[2],
            NX_ID = self.G.ID.shape[1],
            NY_ID = self.G.ID.shape[2],
            NZ_ID = self.G.ID.shape[3],
            NX_T = nx_t,
            NY_T = ny_t,
            NZ_T = nz_t
        )

        # check if the total constant memory exceeds the variable nbytes

        # init gpu arrays
        self.G.cl_initialize_arrays(self.queue)

        # for dispersive materials
        if Material.maxpoles > 0:
            self.G.cl_initialize_dispersive_arrays(self.queue)

        self.memUsage = self.G.clMemoryUsage            

        # if pmls
        if self.G.pmls:
            pmlmodulelectric = 'pml_updates_electric_' + self.G.pmlformulation + '.cl'
            pmlmodulemagnetic = 'pml_updates_magnetic_' + self.G.pmlformulation + '.cl'

            kernel_pml_electric = trad_jinja_env.get_template(pmlmodulelectric).render(
                REAL=self.datatypes['REAL'],
                N_updatecoeffsE=self.G.updatecoeffsE.size, 
                NY_MATCOEFFS=self.G.updatecoeffsE.shape[1], 
                NX_FIELDS=self.G.Ex.shape[0], 
                NY_FIELDS=self.G.Ex.shape[1],
                NZ_FIELDS=self.G.Ex.shape[2], 
                updateEVal = self.updateEVal,
                updateHVal = self.updateHVal,
                NX_ID=self.G.ID.shape[1], 
                NY_ID=self.G.ID.shape[2], 
                NZ_ID=self.G.ID.shape[3]
            )

            kernel_pml_magnetic = trad_jinja_env.get_template(pmlmodulemagnetic).render(
                REAL=self.datatypes['REAL'], 
                N_updatecoeffsH=self.G.updatecoeffsH.size, 
                NY_MATCOEFFS=self.G.updatecoeffsH.shape[1], 
                NX_FIELDS=self.G.Hx.shape[0], 
                NY_FIELDS=self.G.Hx.shape[1], 
                NZ_FIELDS=self.G.Hx.shape[2],
                updateEVal = self.updateEVal,
                updateHVal = self.updateHVal,
                NX_ID=self.G.ID.shape[1], 
                NY_ID=self.G.ID.shape[2], 
                NZ_ID=self.G.ID.shape[3]
            )

            for pml in self.G.pmls:
                pml.cl_set_workgroups(self.G)
                pml.cl_initialize_arrays(self.queue)
                pml.cl_set_program(self.context, kernel_pml_electric, kernel_pml_magnetic)
                self.memUsage += pml.clMemoryUsage

        # if receviers
        if self.G.rxs:
            self.rxcoords_cl, self.rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
            self.memUsage += self.rxcoords_cl.nbytes + self.rxs_cl.nbytes
            # get the store kernel function
            store_output_text = trad_jinja_env.get_template('store_outputs.cl').render(
                REAL=self.datatypes['REAL'], 
                NY_RXCOORDS=3,
                NX_RXS=6, 
                NY_RXS=self.G.iterations, 
                NZ_RXS=len(self.G.rxs), 
                NX_FIELDS=self.G.Ex.shape[0], 
                NY_FIELDS=self.G.Ex.shape[1], 
                NZ_FIELDS=self.G.Ex.shape[2]
            )

        # if sources
        if self.G.voltagesources + self.G.hertziandipoles + self.G.magneticdipoles:
            sources_text = trad_jinja_env.get_template('update_source.cl').render(
                REAL=self.datatypes['REAL'], 
                N_updatecoeffsE=self.G.updatecoeffsE.size, 
                N_updatecoeffsH=self.G.updatecoeffsH.size, 
                updateEVal = self.updateEVal,
                updateHVal = self.updateHVal,
                NY_MATCOEFFS=self.G.updatecoeffsE.shape[1], 
                NY_SRCINFO=4, 
                NY_SRCWAVES=self.G.iterations, 
                NX_FIELDS=self.G.Ex.shape[0],
                NY_FIELDS=self.G.Ex.shape[1], 
                NZ_FIELDS=self.G.Ex.shape[2], 
                NX_ID=self.G.ID.shape[1], 
                NY_ID=self.G.ID.shape[2], 
                NZ_ID=self.G.ID.shape[3]
            )

            if self.G.hertziandipoles:
                self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_hertzian_cl.nbytes + self.srcinfo2_hertzian_cl.nbytes + self.srcwaves_hertzian_cl.nbytes
            if self.G.magneticdipoles:
                self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl, self.srcwaves_magnetic_cl = gpu_initialise_src_arrays(self.G.magneticdipoles, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_magnetic_cl.nbytes + self.srcinfo2_magnetic_cl.nbytes + self.srcwaves_magnetic_cl.nbytes
            if self.G.voltagesources:
                self.srcinfo1_voltage_cl, self.srcinfo2_voltage_cl, self.srcwaves_voltage_cl = gpu_initialise_src_arrays(self.G.voltagesources, self.G, queue=self.queue, opencl=True)
                self.memUsage += self.srcinfo1_voltage_cl.nbytes + self.srcinfo2_voltage_cl.nbytes + self.srcwaves_voltage_cl.nbytes

        if self.G.snapshots:
            self.snapEx_cl, self.snapEy_cl, self.snapEz_cl, self.snapHx_cl, self.snapHy_cl, self.snapHz_cl = cl_initialise_snapshot_array(self.queue, self.G)
            snapshot_text = trad_jinja_env.get_template('snapshots.cl').render(
                REAL=self.datatypes['REAL'], 
                NX_SNAPS=Snapshot.nx_max, 
                NY_SNAPS=Snapshot.ny_max, 
                NZ_SNAPS=Snapshot.nz_max, 
                NX_FIELDS=self.G.Ex.shape[0], 
                NY_FIELDS=self.G.Ex.shape[1], 
                NZ_FIELDS=self.G.Ex.shape[2]
            )
            self.snapshot_prg = cl.Program(self.context, snapshot_text).build()

        self.store_output_prg = cl.Program(self.context, store_output_text).build()
        self.source_prg = cl.Program(self.context, sources_text).build()
        self.kernel_field_prg = cl.Program(self.context, kernel_fields_text).build()

    def checkConstantMem(self):
        # get the constant memory for current device 
        EValMem = self.updateEVal.size*self.updateEVal.itemsize
        HValMem = self.updateHVal.size*self.updateHVal.itemsize
        totalMem = EValMem + HValMem
        if(totalMem >= self.deviceParam['MAX_CONSTANT_BUFFER_SIZE']):
            print("Constant memory insufficient for E/H field update coefficients")
            return False
        else:
            return True

    def clMemoryCheck(self, snapsmemsize):
        # nvidia gpu profilers show overhead memory usage of ~42MB

        # snapsmemsize should be in bytes
        if snapsmemsize!=0:            
            self.memUsage += snapsmemsize

        if self.nvidiaGPU:
            print("Estimated Nvidia GPU Memory to be used %.4f MB"%((self.memUsage/1024/1024) + 42))
        else:
            print("Estimated device memory utilization : %.4f MB"%(self.memUsage/1024/1024))
        
        print("Global Device Memory available is %.4f MB"%(self.deviceParam['GLOBAL_MEM_SIZE']/1024/1024))
        
        if(self.deviceParam['GLOBAL_MEM_SIZE'] <= self.memUsage):
            if self.memUsage-snapsmemsize < self.deviceParam['GLOBAL_MEM_SIZE'] and snapsmemsize!=0:
                # snapgpu2cpu is true, because the model can run if the snapshot things happens at the host level
                self.G.snapsgpu2cpu= True
            else:
                # even if the snapshots are copied back to host the gpu wont fit the model
                print("Exiting simulation due to less memory")
                return False 
        else:
            # the snapshots can accomodate in the device memory itself 
            return True


    def solver(self, currentmodelrun, modelend, G, elementwisekernel=False, snapsmemsize=0):
        """
        This will be the core part of the OpenCl Solver
        
        Args::
            currentmodelrun : current model run number
            modelend : number of last model to run
            G : grid class instance - holds essential parameters describing the model
        Returns::
            tsolve (float) : time taken to execute solving
            memsolve (int) : memory usage on final iteration in bytes
        """
        assert self.platforms is not None
        assert self.devices is not None 

        self.G = G
        self.tsolverstart = timer()
        elapsed = 0

        # create context and command queues
        self.createContext()

        # set types
        self.setDataTypes()

        # set global constant of the update coefficients
        # get the gpu format of the updatecoefficients and add them to the memory
        self.updateEVal = self.G.updatecoeffsE.ravel()
        self.updateHVal = self.G.updatecoeffsH.ravel()

        # add a check for max constant memory check with the update values
        if not self.checkConstantMem():
            print("Exiting Simulations")
            return 0,0

        if elementwisekernel is True:
            print("Building Kernels using pyopencl.elementwise")
            self.elwise_kernel_build()            

            # making memory check with estimated memory usages
            if not self.clMemoryCheck(snapsmemsize):
                return 0,0

            for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbars):

                if self.G.rxs:
                    event = self.store_field(
                        np.int32(len(self.G.rxs)), np.int32(iteration),
                        self.rxcoords_cl, self.rxs_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                # store snapshots

                #update magnetic field
                event = self.update_h_field(
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl, self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl,
                    self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                )
                event.wait()
                elapsed += 1e-9*(event.profile.end - event.profile.start)

                #pmls
                for pml in self.G.pmls:
                    function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                    event = self.pml_magnetic_update[function_name](
                        np.int32(pml.xs), np.int32(pml.xf), np.int32(pml.ys), 
                        np.int32(pml.yf), np.int32(pml.zs), np.int32(pml.zf), 
                        np.int32(pml.HPhi1.shape[1]), np.int32(pml.HPhi1.shape[2]), np.int32(pml.HPhi1.shape[3]), 
                        np.int32(pml.HPhi2.shape[1]), np.int32(pml.HPhi2.shape[2]), np.int32(pml.HPhi2.shape[3]), 
                        np.int32(pml.thickness), self.G.ID_cl, 
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl, 
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl, 
                        pml.HPhi1_cl, pml.HPhi2_cl, 
                        pml.HRA_cl, pml.HRB_cl, pml.HRE_cl, pml.HRF_cl, np.float32(pml.d)
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                
                # magnetic dipoles
                if self.G.magneticdipoles:
                    event = self.magneticdipole_update(
                        np.int32(len(self.G.magneticdipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl,
                        self.srcwaves_magnetic_cl, self.G.ID_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)
                
                # dispersive materials
                if Material.maxpoles > 0:
                    event = self.update_e_dispersive_A(
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl,
                        self.G.Tx_cl, self.G.Ty_cl, self.G.Tz_cl, self.G.ID_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                else:
                    event = self.update_e_field(
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        self.G.ID_cl, self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                #pmls
                for pml in self.G.pmls:
                    function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                    event = self.pml_electric_update[function_name](
                        np.int32(pml.xs), np.int32(pml.xf), np.int32(pml.ys), 
                        np.int32(pml.yf), np.int32(pml.zs), np.int32(pml.zf), 
                        np.int32(pml.EPhi1.shape[1]), np.int32(pml.EPhi1.shape[2]), np.int32(pml.EPhi1.shape[3]), 
                        np.int32(pml.EPhi2.shape[1]), np.int32(pml.EPhi2.shape[2]), np.int32(pml.EPhi2.shape[3]), 
                        np.int32(pml.thickness), self.G.ID_cl, 
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl, 
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl, 
                        pml.EPhi1_cl, pml.EPhi2_cl, 
                        pml.ERA_cl, pml.ERB_cl, pml.ERE_cl, pml.ERF_cl, np.float32(pml.d)
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                if self.G.voltagesources:
                    event = self.voltagesource_update(
                        np.int32(len(self.G.voltagesources)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_voltage_cl, self.srcinfo2_voltage_cl,
                        self.srcwaves_voltage_cl, self.G.ID_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)
                
                if self.G.hertziandipoles:
                    event = self.hertziandipoles_update(
                        np.int32(len(G.hertziandipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl, 
                        self.G.ID_cl, self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)


                #
                if Material.maxpoles > 0:
                    event = self.update_e_dispersive_B(
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl,
                        self.G.Tx_cl, self.G.Ty_cl, self.G.Tz_cl, self.G.ID_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

        else:
            print("Building Kernels the traditional method")
            self.traditional_kernel_build()

            # making memory check with estimated memory usages
            if not self.clMemoryCheck(0):
                return 0,0

            for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbars):

                # store field component values for every receiver
                if self.G.rxs:
                    event = self.store_output_prg.store_outputs(
                        self.queue, (len(self.G.rxs),1,1), None, 
                        np.int32(len(self.G.rxs)), np.int32(iteration), 
                        self.rxcoords_cl.data, self.rxs_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                # store any snapshots
                for i, snap in enumerate(self.G.snapshots):
                    if snap.time == iteration + 1:
                        if not self.G.snapsgpu2cpu:
                            event = self.snapshot_prg.store_snapshot(
                                self.queue, Snapshot.cl_workgroup, None,
                                np.int32(i), np.int32(snap.xs), np.int32(snap.xf),
                                np.int32(snap.ys), np.int32(snap.yf), 
                                np.int32(snap.zs), np.int32(snap.zf),
                                np.int32(snap.dy), np.int32(snap.dz),
                                self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                                self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data,
                                self.snapEx_cl.data, self.snapEy_cl.data, self.snapEz_cl.data,
                                self.snapHx_cl.data, self.snapHy_cl.data, self.snapHz_cl.data
                            )
                            event.wait()
                            elapsed += 1e-9*(event.profile.end - event.profile.start)

                        else:
                            event = self.snapshot_prg.store_snapshot(
                                self.queue, Snapshot.cl_workgroup, None,
                                np.int32(0), np.int32(snap.xs), np.int32(snap.xf),
                                np.int32(snap.ys), np.int32(snap.yf), 
                                np.int32(snap.zs), np.int32(snap.zf),
                                np.int32(snap.dy), np.int32(snap.dz),
                                self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                                self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data,
                                self.snapEx_cl.data, self.snapEy_cl.data, self.snapEz_cl.data,
                                self.snapHx_cl.data, self.snapHy_cl.data, self.snapHz_cl.data
                            )
                            event.wait()
                            elapsed += 1e-9*(event.profile.end - event.profile.start)
                            
                            gpu_get_snapshot_array(self.snapEx_cl.get(), self.snapEy_cl.get(), self.snapEz_cl.get(), 
                                                   self.snapHx_cl.get(), self.snapHy_cl.get(), self.snapHz_cl.get(), 0, snap)
                            
                # update magnetic field components 
                event = self.kernel_field_prg.update_h(
                    self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl.data, self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data, 
                    self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                )
                event.wait()
                elapsed += 1e-9*(event.profile.end - event.profile.start)

                for pml in self.G.pmls:
                    pml.cl_update_magnetic(self.queue, self.G)
                    elapsed += pml.elapsed

                # update magnetic dipoles (sources)
                if self.G.magneticdipoles:
                    event = self.source_prg.update_magnetic_dipole(
                        self.queue, (len(self.G.magneticdipoles),1,1), None,
                        np.int32(len(self.G.magneticdipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_magnetic_cl.data, self.srcinfo2_magnetic_cl.data,
                        self.srcwaves_magnetic_cl.data, self.G.ID_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                if Material.maxpoles > 0:
                    event = self.kernel_field_prg.update_e_dispersive_A(
                        self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl.data,
                        self.G.Tx_cl.data, self.G.Ty_cl.data, self.G.Tz_cl.data, self.G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                else:
                    # update electric field components
                    event = self.kernel_field_prg.update_e(
                        self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                for pml in self.G.pmls:
                    pml.cl_update_electric(self.queue, self.G)
                    elapsed += pml.elapsed
                
                
                if self.G.voltagesources:
                    event = self.source_prg.update_voltage_source(
                        self.queue, (len(self.G.voltagesources),1,1), None,
                        np.int32(len(self.G.voltagesources)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_voltage_cl.data, self.srcinfo2_voltage_cl.data,
                        self.srcwaves_voltage_cl.data, self.G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

                if self.G.hertziandipoles:
                    event = self.source_prg.update_hertzian_dipole(
                        self.queue, (len(self.G.hertziandipoles),1,1), None,
                        np.int32(len(self.G.hertziandipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_hertzian_cl.data, self.srcinfo2_hertzian_cl.data, self.srcwaves_hertzian_cl.data, 
                        self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)


                if Material.maxpoles > 0:
                    event = self.kernel_field_prg.update_e_dispersive_B(
                        self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl.data,
                        self.G.Tx_cl.data, self.G.Ty_cl.data, self.G.Tz_cl.data, self.G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                    )
                    event.wait()
                    elapsed += 1e-9*(event.profile.end - event.profile.start)

        if self.G.rxs:
            # store the output from receivers array back to correct receiver objects
            gpu_get_rx_array(self.rxs_cl.get(), self.rxcoords_cl.get(), self.G)

        # copy data from any snapshots back to correct snapshot objects
        if self.G.snapshots and not self.G.snapsgpu2cpu:
            for i, snap in enumerate(self.G.snapshots):
                gpu_get_snapshot_array(self.snapEx_cl.get(), self.snapEy_cl.get(), self.snapEz_cl.get(),
                                       self.snapHx_cl.get(), self.snapHy_cl.get(), self.snapHz_cl.get(), i, snap)

        print("Time reported from the timer() module is : %.6f"%(timer()-self.tsolverstart))
            
        return elapsed,(self.memUsage/1024/1024)