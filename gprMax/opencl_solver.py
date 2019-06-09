import numpy as np 
import os 
import sys 
import tqdm
import warnings
import time 

import jinja2
from jinja2 import Template
import pyopencl as cl
import pyopencl.array as cl_array

from gprMax.materials import Material, process_materials
from gprMax.receivers import gpu_initialise_rx_arrays, gpu_get_rx_array
from gprMax.sources import gpu_initialise_src_arrays

class OpenClSolver(object):
    def __init__(self, G=None, context=None, queue=None):
        self.context = context
        self.queue = queue
        self.G = G
        self.jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))

    def setDeviceParameters(self):
        pass

    def getPlatformNDevices(self, platformIdx=1, deviceIdx=1):
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
            'REAL': 'float'
        }
        return

    # def cl_initialize_arrays(self):
    #     self.ID_cl = cl_array.to_device(self.queue, self.G.ID)
    #     self.Ex_cl = cl_array.to_device(self.queue, self.G.Ex)
    #     self.Ey_cl = cl_array.to_device(self.queue, self.G.Ey)
    #     self.Ez_cl = cl_array.to_device(self.queue, self.G.Ez)
    #     self.Hx_cl = cl_array.to_device(self.queue, self.G.Hx)
    #     self.Hy_cl = cl_array.to_device(self.queue, self.G.Hy)
    #     self.Hz_cl = cl_array.to_device(self.queue, self.G.Hz)

    def solver(self, currentmodelrun, modelend, G):
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
        
        # get the devices and platforms
        # if (self.queue and self.context) is None:
        #     self.getPlatformNDevices()
        assert self.platforms is not None
        assert self.devices is not None 
        
        self.G = G

        # create context and command queues
        self.createContext()

        # set types
        self.setDataTypes()

        # for materials
        if Material.maxpoles > 0:
            raise NotImplementedError
        else:
            # set parameters accordingly
            warnings.warn('see the parameters accordingly')
            kernel_fields_text = self.jinja_env.get_template('update_fields.cl').render(
                REAL = self.datatypes['REAL'],
                N_updatecoeffsE = self.G.updatecoeffsE.size,
                N_updatecoeffsH = self.G.updatecoeffsH.size,
                NY_MATCOEFFS = self.G.updatecoeffsE.shape[1],
                NY_MATDISPCOEFFS = 1,
                NX_FIELDS = self.G.Ex.shape[0],
                NY_FEILDS = self.G.Ex.shape[1],
                NZ_FIELDS = self.G.Ex.shape[2],
                NX_ID = self.G.ID.shape[1],
                NY_ID = self.G.ID.shape[2],
                NZ_ID = self.G.ID.shape[3],
                NX_T = 1,
                NY_T = 1,
                NZ_T = 1
            )

        # set global constant of the update coefficients
        # get the gpu format of the updatecoefficients and add them to the memory
        self.updatecoeffsE = cl_array.to_device(self.queue, self.G.updatecoeffsE)
        self.updatecoeffsH = cl_array.to_device(self.queue, self.G.updatecoeffsH)

        warnings.warn("transfer the memory to the device using the kernel, also check the memory")

        # check for dispersive materials / if so then get kernel func and init the dispersive gpu arrays
        if Material.maxpoles > 0:
            raise NotImplementedError

        # set blocks per grid and init the gpu arrays
        self.G.cl_initialize_arrays(self.queue)

        # if pmls
        if self.G.pmls:
            warnings.warn("PML not implemented")
            pass 

        # if receivers
        if self.G.rxs:
            rxcoords_cl, rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
            
            # get the store kernel function 
            store_output_text = self.jinja_env.get_template('store_outputs.cl').render(
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
            # print(self.G.voltagesources, self.G.hertziandipoles, self.G.magneticdipoles)

            # get the kernel function for the sources
            sources_text = self.jinja_env.get_template('update_source.cl').render(
                REAL=self.datatypes['REAL'], 
                N_updatecoeffsE=self.G.updatecoeffsE.size, 
                N_updatecoeffsH=self.G.updatecoeffsH.size, 
                NY_MATCOEFFS=self.G.updatecoeffsE.shape[1], 
                NY_SRCINFO=4, NY_SRCWAVES=self.G.iterations, 
                NX_FIELDS=self.G.Ex.shape[0],
                NY_FIELDS=self.G.Ex.shape[1], 
                NZ_FIELDS=self.G.Ex.shape[2], 
                NX_ID=self.G.ID.shape[1], 
                NY_ID=self.G.ID.shape[2], 
                NZ_ID=self.G.ID.shape[3]
            )
            

            if self.G.hertziandipoles:
                srcinfo1_hertzian_cl, srcinfo2_hertzian_cl, srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
                # update_hertzian_dipole_text = self.jinja_env.get_template('update_source.cl')
            if self.G.magneticdipoles:
                raise NotImplementedError
            if self.G.voltagesources:
                raise NotImplementedError

        if self.G.snapshots:
            raise NotImplementedError


        kernel_field_prg = cl.Program(self.context, kernel_fields_text).build()
        # copy the updatecoefficients to the device for all the necessary kernels
        # field constant
        kernel_field_prg.setUpdateCoeffs(
            self.queue, (1,1,1), None, self.updatecoeffsE.data, self.updatecoeffsH.data
        )

        source_prg = cl.Program(self.context, sources_text).build()

        # get the global values ???
        # the self object of the constant values has to be passed to the prg.kernel name

        source_prg.setUpdateCoeffs(
            self.queue, (1,1,1), None, self.updatecoeffsE.data, self.updatecoeffsH.data
        )

        store_output_prg = cl.Program(self.context, store_output_text).build()

        for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbar):

            # get gpu memory

            # store field component values for every receiver
            if self.G.rxs:
                warnings.warn("Have to set up the global and local size")
                store_output_prg.store_outputs(
                    self.queue, (1,1,1), None, 
                    np.int32(len(self.G.rxs)), np.int32(iteration), 
                    rxcoords_cl.data, rxs_cl.data, 
                    self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                    self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                )

            # store any snapshots

            # update magnetic field components 
            warnings.warn("update the cl arrays accordingly")
            kernel_field_prg.update_h(
                self.queue, (256,1,1), None, 
                np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                self.G.ID_cl.data, self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data, 
                self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
            )

            for pml in self.G.pmls:
                warnings.warn("Not implemented as of now")
                pass

            # update magnetic dipoles (sources)
            if self.G.magneticdipoles:
                raise NotImplementedError

            if Material.maxpoles != 0:
                raise NotImplementedError
            else:
                # update electric field components
                warnings.warn("update the cl arrays accordingly")
                kernel_field_prg.update_e(
                    self.queue, (256,1,1), None, 
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                    self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                )

            for pml in self.G.pmls:
                warnings.warn("Not implemented as of now")
                pass
            

            if self.G.voltagesources:
                raise NotImplementedError

            if self.G.hertziandipoles:
                source_prg.update_hertzian_dipole(
                    self.queue, (1,1,1), None,
                    np.int32(len(self.G.hertziandipoles)), np.int32(iteration), 
                    srcinfo1_hertzian_cl.data, srcinfo2_hertzian_cl.data, srcwaves_hertzian_cl.data, 
                    self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                )

            if Material.maxpoles > 0:
                raise NotImplementedError

        
        if self.G.rxs:
            # store the output from receivers array back to corrent receiver objects
            gpu_get_rx_array(rxs_cl.get(), rxcoords_cl.get(), self.G)
            
        # copy data from any snapshots back to correct snapshot objects
        if self.G.snapshots:
            raise NotImplementedError


        # close context and queues
        return 1,2


                







    
