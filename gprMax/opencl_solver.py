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
from gprMax.utilities import get_terminal_width, round32
from gprMax.opencl_el_kernels import pml_updates_electric_HORIPML as el_electric
from gprMax.opencl_el_kernels import pml_updates_magnetic_HORIPML as el_magnetic

class OpenClSolver(object):
    def __init__(self, G=None, context=None, queue=None):
        self.context = context
        self.queue = queue
        self.G = G
        self.jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))

    def setDeviceParameters(self):
        pass

    def getPlatformNDevices(self, platformIdx=None, deviceIdx=None):
        print("")
        self.platforms = cl.get_platforms() 

        print("Following platform supporting OpenCl were discovered")
        for idx, platf in enumerate(self.platforms):
            print("{} Platform: {}".format(str(idx), str(platf.name)))
        if platformIdx is None:
            self.platformIdx = int(input("Platform to be chosen ?"))
            assert self.platformIdx in [i for i in range(len(self.platforms))]
        else:
            self.platformIdx = platformIdx

        print("Following devices supporting OpenCl were discovered for platform: {}".format(str(self.platforms[self.platformIdx].name)))
        self.devices = self.platforms[self.platformIdx].get_devices()
        for idx, dev in enumerate(self.devices):
            print("{} Devices: {}".format(str(idx), str(dev.name)))
        
        if deviceIdx is None: 
            self.deviceIdx = int(input("Device to be chosen?"))
            assert self.deviceIdx in [i for i in range(len(self.devices))]
        else:
            self.deviceIdx = deviceIdx

        print("Chosen Platform and Device")
        print("Platform: {}".format(str(self.platforms[self.platformIdx].name)))
        print("Devices: {}".format(str(self.devices[self.deviceIdx].name)))

        return

    def createContext(self):
        if self.context is None:
            print("Creating context...")
            self.context = cl.Context(devices=[self.devices[self.deviceIdx]])
        else:
            pass 

        if self.queue is None:
            print("Creating the command queue...")
            self.queue = cl.CommandQueue(self.context)
        else: 
            pass 

        return
    
    def setDataTypes(self):
        self.datatypes = {
            'REAL': 'float',
            'COMPLEX': 'cfloat'
        }
        return

    def setKernelParameters(self):
        pass

    def elwise_kernel_build(self):
        
        elwise_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_el_kernels'))
        
        # get the preamble
        common_kernel = elwise_jinja_env.get_template('common.cl').render(
            REAL = self.datatypes['REAL'],
            N_updatecoeffsE = self.G.updatecoeffsE.size,
            N_updatecoeffsH = self.G.updatecoeffsH.size,
            updateEVal = self.updateEVal,
            updateHVal = self.updateHVal,
            NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS = 1,
            NX_FIELDS = self.G.Ex.shape[0],
            NY_FIELDS = self.G.Ex.shape[1],
            NZ_FIELDS = self.G.Ex.shape[2],
            NX_ID = self.G.ID.shape[1],
            NY_ID = self.G.ID.shape[2],
            NZ_ID = self.G.ID.shape[3],
            NX_T = 1,
            NY_T = 1,
            NZ_T = 1,
            NY_RXCOORDS=3,
            NX_RXS=6, 
            NY_RXS=self.G.iterations, 
            NZ_RXS=len(self.G.rxs),
            NY_SRCINFO=4, 
            NY_SRCWAVES=self.G.iterations
        )

        if Material.maxpoles > 0:
            raise NotImplementedError
        else:
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

        # check for dispersive materials
        if Material.maxpoles > 0:
            raise NotImplementedError

        # initialize the cl arrays
        self.G.cl_initialize_arrays(self.queue)

        # if pmls 
        if self.G.pmls:

            # get the electric and magnetic kernel files
            self.pml_magnetic_update = {}
            self.pml_electric_update = {}

            for pml in self.G.pmls:
                pml.cl_set_workgroups(self.G)
                pml.cl_initialize_arrays(self.queue)
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
                    'order2_xminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order1_xplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global $REAL *Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_xplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order1_yminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_yminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order1_yplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global $REAL *Ex, __global const $REAL* restrict Ey, __global $REAL *Ez, __global const $REAL* restrict Hx, __global const $REAL* restrict Hy, __global const $REAL* restrict Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    'order2_yplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order1_zminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order2_zminus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order1_zplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    'order2_zplus' : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d"
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
                    "order2_xminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global {{REAL}} *Hy, __global {{REAL}} *Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order1_xplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global const $REAL* restrict Hx, __global $REAL *Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_xplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global {{REAL}} *Hy, __global {{REAL}} *Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order1_yminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_yminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global const {{REAL}}* restrict Hy, __global {{REAL}} *Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order1_yplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const $REAL* restrict Ex, __global const $REAL* restrict Ey, __global const $REAL* restrict Ez, __global $REAL *Hx, __global const $REAL* restrict Hy, __global $REAL *Hz, __global $REAL *PHI1, __global $REAL *PHI2, __global const $REAL* restrict RA, __global const $REAL* restrict RB, __global const $REAL* restrict RE, __global const $REAL* restrict RF, $REAL d",
                    "order2_yplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global const {{REAL}}* restrict Hy, __global {{REAL}} *Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order1_zminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global {{REAL}} *Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order2_zminus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global {{REAL}} *Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order1_zplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global {{REAL}} *Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d",
                    "order2_zplus" : "int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global const {{REAL}}* restrict Ey, __global const {{REAL}}* restrict Ez, __global {{REAL}} *Hx, __global {{REAL}} *Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d"
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

        if self.G.snapshots:
            raise NotImplementedError


    def traditional_kernel_build(self):
        # set the jinja engine
        trad_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))
        
        # for field update
        if Material.maxpoles > 0:
            raise NotImplementedError
        else:
            kernel_fields_text = trad_jinja_env.get_template('update_fields.cl').render(
                REAL = self.datatypes['REAL'],
                COMPLEX = self.datatypes['COMPLEX'],
                N_updatecoeffsE = self.G.updatecoeffsE.size,
                N_updatecoeffsH = self.G.updatecoeffsH.size,
                updateEVal = self.updateEVal,
                updateHVal = self.updateHVal,
                NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
                NY_MATDISPCOEFFS = 1,
                NX_FIELDS = self.G.Ex.shape[0],
                NY_FIELDS = self.G.Ex.shape[1],
                NZ_FIELDS = self.G.Ex.shape[2],
                NX_ID = self.G.ID.shape[1],
                NY_ID = self.G.ID.shape[2],
                NZ_ID = self.G.ID.shape[3],
                NX_T = 1,
                NY_T = 1,
                NZ_T = 1
            )

        # check if the total constant memory exceeds the variable nbytes

        # for dispersive materials
        if Material.maxpoles > 0:
            raise NotImplementedError

        # init gpu arrays
        self.G.cl_initialize_arrays(self.queue)

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

        # if receviers
        if self.G.rxs:
            self.rxcoords_cl, self.rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
            
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
            if self.G.magneticdipoles:
                self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl, self.srcwaves_magnetic_cl = gpu_initialise_src_arrays(self.G.magneticdipoles, self.G, queue=self.queue, opencl=True)
            if self.G.voltagesources:
                self.srcinfo1_voltage_cl, self.srcinfo2_voltage_cl, self.srcwaves_voltage_cl = gpu_initialise_src_arrays(self.G.voltagesources, self.G, queue=self.queue, opencl=True)

        if self.G.snapshots:
            raise NotImplementedError

        self.store_output_prg = cl.Program(self.context, store_output_text).build()
        self.source_prg = cl.Program(self.context, sources_text).build()
        self.kernel_field_prg = cl.Program(self.context, kernel_fields_text).build()


    def solver(self, currentmodelrun, modelend, G, elementwisekernel=False):
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

        # create context and command queues
        self.createContext()

        # set types
        self.setDataTypes()

        # set global constant of the update coefficients
        # get the gpu format of the updatecoefficients and add them to the memory
        self.updateEVal = self.G.updatecoeffsE.ravel()
        self.updateHVal = self.G.updatecoeffsH.ravel()

        if elementwisekernel is True:
            print("Building Kernels using pyopencl.elementwise")
            self.elwise_kernel_build()

            for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbars):  
                
                # get memory info and do memory checks

                if self.G.rxs:
                    self.store_field(
                        np.int32(len(self.G.rxs)), np.int32(iteration),
                        self.rxcoords_cl, self.rxs_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                
                # store snapshots

                #update magnetic field
                self.update_h_field(
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl, self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl,
                    self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                )

                #pmls
                for pml in self.G.pmls:
                    function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                    self.pml_magnetic_update[function_name](
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
                
                # magnetic dipoles
                if self.G.magneticdipoles:
                    self.magneticdipole_update(
                        np.int32(len(self.G.magneticdipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl,
                        self.srcwaves_magnetic_cl, self.G.ID_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )
                
                # dispersive materials
                if Material.maxpoles != 0:
                    raise NotImplementedError
                else:
                    self.update_e_field(
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        self.G.ID_cl, self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl,
                        self.G.Hx_cl, self.G.Hy_cl, self.G.Hz_cl
                    )

                #pmls
                for pml in self.G.pmls:
                    function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                    self.pml_electric_update[function_name](
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

                if self.G.voltagesources:
                    self.voltagesource_update(
                        np.int32(len(self.G.voltagesources)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_voltage_cl, self.srcinfo2_voltage_cl,
                        self.srcwaves_voltage_cl, self.G.ID_cl,
                        self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                    )
                
                if self.G.hertziandipoles:
                    self.hertziandipoles_update(
                        np.int32(len(G.hertziandipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl, 
                        self.G.ID_cl, self.G.Ex_cl, self.G.Ey_cl, self.G.Ez_cl
                    )

                #
                if Material.maxpoles > 0:
                    raise NotImplementedError

        else:
            print("Building Kernels the traditional method")
            self.traditional_kernel_build()

            for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbars):

                # get gpu memory

                # store field component values for every receiver
                if self.G.rxs:
                    store_output_event = self.store_output_prg.store_outputs(
                        self.queue, (len(self.G.rxs),1,1), None, 
                        np.int32(len(self.G.rxs)), np.int32(iteration), 
                        self.rxcoords_cl.data, self.rxs_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    store_output_event.wait()

                # store any snapshots

                # update magnetic field components 
                kernel_field_event = self.kernel_field_prg.update_h(
                    self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl.data, self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data, 
                    self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                )
                kernel_field_event.wait()

                for pml in self.G.pmls:
                    pml.cl_update_magnetic(self.queue, self.G)            

                # update magnetic dipoles (sources)
                if self.G.magneticdipoles:
                    source_event = self.source_prg.update_magnetic_dipole(
                        self.queue, (1,1,1), None,
                        np.int32(len(self.G.magneticdipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_magnetic_cl.data, self.srcinfo2_magnetic_cl.data,
                        self.srcwaves_magnetic_cl.data, self.G.ID_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    source_event.wait()

                if Material.maxpoles != 0:
                    raise NotImplementedError
                else:
                    # update electric field components
                    kernel_field_event = self.kernel_field_prg.update_e(
                        self.queue, (int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    kernel_field_event.wait()

                for pml in self.G.pmls:
                    pml.cl_update_electric(self.queue, self.G)
                
                
                if self.G.voltagesources:
                    source_event = self.source_prg.update_voltage_source(
                        self.queue, (1,1,1), None,
                        np.int32(len(self.G.voltagesources)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_voltage_cl.data, self.srcinfo2_voltage_cl.data,
                        self.srcwaves_voltage_cl.data, self.G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                    )
                    source_event.wait()

                if self.G.hertziandipoles:
                    source_event = self.source_prg.update_hertzian_dipole(
                        self.queue, (1,1,1), None,
                        np.int32(len(self.G.hertziandipoles)), np.int32(iteration),
                        np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                        self.srcinfo1_hertzian_cl.data, self.srcinfo2_hertzian_cl.data, self.srcwaves_hertzian_cl.data, 
                        self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                    )
                    source_event.wait()

                if Material.maxpoles > 0:
                    raise NotImplementedError

        if self.G.rxs:
            # store the output from receivers array back to correct receiver objects
            gpu_get_rx_array(self.rxs_cl.get(), self.rxcoords_cl.get(), self.G)

        # copy data from any snapshots back to correct snapshot objects
        if self.G.snapshots:
            raise NotImplementedError

        # close context and queues
        return 1,2