import numpy as np 
import os 
import sys 
from tqdm import tqdm
import warnings
import time 

import jinja2
from jinja2 import Template
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

from gprMax.materials import Material, process_materials
from gprMax.receivers import gpu_initialise_rx_arrays, gpu_get_rx_array
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.utilities import get_terminal_width, round32
from gprMax.opencl_el_kernels import pml_updates_electric_HORIPML, pml_updates_magnetic_HORIPML

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
        print("getting the common kernels")
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
            print("setting up kernel field text")
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
                "int NX, int NY, int NZ, __global const unsigned int* restrict ID, __global float *Ex, __global float *Ey, __global float *Ez, __global const float * restrict Hx, __global const float * restrict Hy, __global const float * restrict Hz",
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
                "int NX, int NY, int NZ, __global const unsigned int* restrict ID, __global float *Hx, __global float *Hy, __global float *Hz, __global const float* restrict Ex, __global const float* restrict Ey, __global const float* restrict Ez",
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
            print("Setting up the PMLs for elementwise kernel execution")

            pmlmodulelectric = 'pml_updates_electric_' + self.G.pmlformulation + '.cl'
            pmlmodulemagnetic = 'pml_updates_magnetic_' + self.G.pmlformulation + '.cl'

            # get the electric and magnetic kernel files

            for pml in self.G.pmls:
                pml.cl_set_workgroups(self.G)
                pml.cl_initialize_arrays(self.queue)
                function_name = 'order'+str(len(pml.CFS)) + '_' + pml.direction
                
                electric_context_dict = {
                    'order1_xminus' : pml_updates_electric_HORIPML.order1_xminus,
                    'order1_xplus' : pml_updates_electric_HORIPML.order1_xplus,
                    'order1_yminus' : pml_updates_electric_HORIPML.order1_yminus,
                    'order1_yplus' : pml_updates_electric_HORIPML.order1_yplus
                }
                pml.update_electric_cl = ElementwiseKernel(
                    self.context,
                    "",
                    electric_context_dict[function_name],
                    "pml_updates_electric_HORIPML_{}".format(function_name),
                    preamble=common_kernel
                )

                magnetic_context_dict = {
                    'order1_xminus' : pml_updates_magnetic_HORIPML.order1_xminus,
                    'order1_xplus' : pml_updates_magnetic_HORIPML.order1_xplus,
                    'order1_yminus' : pml_updates_magnetic_HORIPML.order1_yminus,
                    'order1_yplus': pml_updates_magnetic_HORIPML.order1_yplus
                }
                pml.update_magnetic_cl = ElementwiseKernel(
                    self.context,
                    "",
                    magnetic_context_dict[function_name],
                    "pml_updates_magnetic_HORIPML_{}".format(function_name),
                    preamble=common_kernel
                )

                # pml.cl_get_update_funcs(self.context, kernel_pml_electric, kernel_pml_magnetic)


        if self.G.rxs:
            print("setting up the storage text")
            self.rxcoords_cl, self.rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
            store_field_context = elwise_jinja_env.get_template('store_outputs.cl').render()
            self.store_field = ElementwiseKernel(
                self.context,
                "int NRX, int iteration, __global const int* restrict rxcoords, __global float *rxs, __global const float* restrict Ex, __global const float* restrict Ey, __global const float* restrict Ez, __global const float* restrict Hx, __global const float* restrict Hy, __global const float* restrict Hz",
                store_field_context,
                "store_field",
                preamble=common_kernel
            )

        if self.G.voltagesources + self.G.hertziandipoles + self.G.magneticdipoles:
            print("setting the sources context")
            
            if self.G.hertziandipoles:
                print("setting hertziandipoles")
                hertziandipoles_context = elwise_jinja_env.get_template('update_source.cl').render(REAL=self.datatypes['REAL'])
                self.hertziandipoles_update = ElementwiseKernel(
                    self.context,
                    "int NHERTZDIPOLE, int iteration, float dx, float dy, float dz, __global const int* restrict srcinfo1, __global const float* restrict srcinfo2, __global const float* restrict srcwaveforms, __global const unsigned int* restrict ID, __global float *Ex, __global float *Ey, __global float *Ez",
                    hertziandipoles_context,
                    "hertziandipoles_update",
                    preamble=common_kernel
                )
                self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
            if self.G.voltagesources:
                raise NotImplementedError
            if self.G.magneticdipoles:
                raise NotImplementedError

        if self.G.snapshots:
            raise NotImplementedError


    def traditional_kernel_build(self):
        # set the jinja engine
        trad_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))
        
        # for field update
        if Material.maxpoles > 0:
            print("setting up kernel field for dispersive text")
            kernel_fields_text = trad_jinja_env.get_template('update_fields.cl').render(
                REAL = self.datatypes['REAL'],
                COMPLEX = self.datatypes['COMPLEX'],
                N_updatecoeffsE = self.G.updatecoeffsE.size,
                N_updatecoeffsH = self.G.updatecoeffsH.size,
                updateEVal = self.updateEVal,
                updateHVal = self.updateHVal,
                NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
                NY_MATDISPCOEFFS = self.G.updatecoeffsdispersive.shape[1],
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
        else:
            print("setting up kernel field text")
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
            # update_e_dispersive_A_cl = 
            # update_e_dispersive_B_cl = 
            # self.G.cl_initialize_dispersive_arrays()
            raise NotImplementedError

        # init gpu arrays
        self.G.cl_initialize_arrays(self.queue)

        # if pmls
        if self.G.pmls:
            print("setting up the pmls")
            
            # pmlmodulelectric = 'pml_updates_electric_' + self.G.pmlformulation + '.cl'
            # pmlmodulemagnetic = 'pml_updates_magnetic_' + self.G.pmlformulation + '.cl'
            pmlmodulelectric = 'pml_electric.cl'
            pmlmodulemagnetic = 'pml_magnetic.cl'
            print(pmlmodulelectric, pmlmodulemagnetic)
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
                # pml.cl_get_update_funcs(self.context, kernel_pml_electric, kernel_pml_magnetic)

        # if receviers
        if self.G.rxs:
            print("setting up store output text")
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
            print("setting up sources text")
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
                print("Setting up Hertzian Dipole")
                self.srcinfo1_hertzian_cl, self.srcinfo2_hertzian_cl, self.srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
            if self.G.magneticdipoles:
                print("setting up Magnetic Dipoles")
                self.srcinfo1_magnetic_cl, self.srcinfo2_magnetic_cl, self.srcwaves_magnetic_cl = gpu_initialise_src_arrays(self.G.magneticdipoles, self.G, queue=self.queue, opencl=True)
            if self.G.voltagesources:
                print("setting up Volatge Sources")
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
        print(currentmodelrun)
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
                    raise NotImplementedError
                
                # magnetic dipoles
                if self.G.magneticdipoles:
                    raise NotImplementedError
                
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
                    raise NotImplementedError
                
                if self.G.voltagesources:
                    raise NotImplementedError
                
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
                        # self.queue, (round32(len(self.G.rxs)),1,1), None, 
                        self.queue, (1,1,1), None,
                        np.int32(len(self.G.rxs)), np.int32(iteration), 
                        self.rxcoords_cl.data, self.rxs_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    store_output_event.wait()

                # store any snapshots

                # update magnetic field components 
                kernel_field_event = self.kernel_field_prg.update_h(
                    self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
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
                    # raise NotImplementedError
                    kernel_field_event = self.kernel_field_prg.update_e_dispersive_A_cl(
                        self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl.data,
                        self.G.Tx_cl.data, self.G.Ty_cl.data, self.G.Tz_cl.data, self.G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    kernel_field_event.wait()
                else:
                    # update electric field components
                    kernel_field_event = self.kernel_field_prg.update_e(
                        self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                        self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                    )
                    kernel_field_event.wait()

                for pml in self.G.pmls:
                    pml.cl_update_electric(self.queue, self.G)
                
                # print("Size of Ez_cl: ",self.G.Ez_cl.shape)
                
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
                    kernel_field_event = self.kernel_field_prg.update_e_dispersive_B_cl(
                        self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                        np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                        np.int32(Material.maxpoles), self.G.updatecoeffsdispersive_cl.data,
                        self.G.Tx_cl.data, self.G.Ty_cl.data, self.G.Tz_cl.data, G.ID_cl.data,
                        self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                    )
                    kernel_field_event.wait()

        if self.G.rxs:
            # store the output from receivers array back to correct receiver objects
            gpu_get_rx_array(self.rxs_cl.get(), self.rxcoords_cl.get(), self.G)
        # print(rxs_cl.get())
        # print(rxcoords_cl.get())
        # copy data from any snapshots back to correct snapshot objects
        if self.G.snapshots:
            raise NotImplementedError

        # close context and queues
        return 1,2