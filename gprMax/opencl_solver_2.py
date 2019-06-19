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

from gprMax.materials import Material, process_materials
from gprMax.receivers import gpu_initialise_rx_arrays, gpu_get_rx_array
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.utilities import get_terminal_width, round32

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
            'REAL': 'float'
        }
        return

    def elwise_kernel_build(self):
        
        elwise_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_el_kernels'))
        
        # get the preamble
        print("getting the common kernels")
        self.common_kernel = self.elwise_jinja_env.get_template('common.cl').render(
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
                
            )







        pass

    def traditional_kernel_build(self):
        
        trad_jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'opencl_kernels'))
        if Material.maxpoles > 0:
            raise NotImplementedError
        else:
            print("setting up kernel field text")
            kernel_fields_text = trad_jinja_env.get_template('update_fields.cl').render(
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
                NZ_T = 1
            )

        # for dispersive materials
        if Material.maxpoles > 0:
            raise NotImplementedError

        # init gpu arrays
        self.G.cl_initialize_arrays(self.queue)

        # if pmls
        if self.G.pmls:
            raise NotImplementedError

        # if receviers
        if self.G.rxs:
            print("setting up store output text")
            rxcoords_cl, rxs_cl = gpu_initialise_rx_arrays(self.G, self.queue, opencl=True)
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
                srcinfo1_hertzian_cl, srcinfo2_hertzian_cl, srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
            if self.G.magneticdipoles:
                raise NotImplementedError
            if self.G.voltagesources:
                raise NotImplementedError

        if self.G.snapshots:
            raise NotImplementedError

        store_output_prg = cl.Program(self.context, store_output_text).build()
        source_prg = cl.Program(self.context, sources_text).build()
        kernel_field_prg = cl.Program(self.context, kernel_fields_text).build()







        


        pass



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
        print(currentmodelrun)
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

        # set global constant of the update coefficients
        # get the gpu format of the updatecoefficients and add them to the memory
        self.updateEVal = self.G.updatecoeffsE.ravel()
        self.updateHVal = self.G.updatecoeffsH.ravel()

        if config.elementwisekernel is True:
            print("Building Kernels using pyopencl.elementwise")
            self.elwise_kernel_build()
        else:
            print("Building Kernels the traditional method")
            self.traditional_kernel_build()

        # for materials
        if Material.maxpoles > 0:
            raise NotImplementedError
        else:
            print("Setting up kernel field text")
            # set parameters accordingly
            kernel_fields_text = self.jinja_env.get_template('update_fields.cl').render(
                REAL = self.datatypes['REAL'],
                N_updatecoeffsE = self.G.updatecoeffsE.size,
                N_updatecoeffsH = self.G.updatecoeffsH.size,
                updateEVal = updateEVal,
                updateHVal = updateHVal,
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

        
        # print("G update coeffs : ", G.updatecoeffsH)
        # print("GPU update coeffs : ", self.updatecoeffsH.get())

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
            print("Setting up store output text")
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
            print("Setting up sources text")
            # print(self.G.voltagesources, self.G.hertziandipoles, self.G.magneticdipoles)

            # get the kernel function for the sources
            sources_text = self.jinja_env.get_template('update_source.cl').render(
                REAL=self.datatypes['REAL'], 
                N_updatecoeffsE=self.G.updatecoeffsE.size, 
                N_updatecoeffsH=self.G.updatecoeffsH.size, 
                updateEVal = updateEVal,
                updateHVal = updateHVal,
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
                srcinfo1_hertzian_cl, srcinfo2_hertzian_cl, srcwaves_hertzian_cl = gpu_initialise_src_arrays(self.G.hertziandipoles, self.G, queue=self.queue, opencl=True)
                print("Source Waveform Values")
                print(srcwaves_hertzian_cl.shape)
                # update_hertzian_dipole_text = self.jinja_env.get_template('update_source.cl')
            if self.G.magneticdipoles:
                raise NotImplementedError
            if self.G.voltagesources:
                raise NotImplementedError

        if self.G.snapshots:
            raise NotImplementedError

        
        store_output_prg = cl.Program(self.context, store_output_text).build()
        source_prg = cl.Program(self.context, sources_text).build()
        kernel_field_prg = cl.Program(self.context, kernel_fields_text).build()


        for iteration in tqdm(range(self.G.iterations), desc="Running simulation model" + str(currentmodelrun) + "/" + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not self.G.progressbars):

            # get gpu memory

            # store field component values for every receiver
            if self.G.rxs:
                # print("In receive point")
                # outkernel  = store_output_prg.store_outputs
                # print(outkernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.devices[self.deviceIdx]))

                store_output_event = store_output_prg.store_outputs(
                    # self.queue, (round32(len(self.G.rxs)),1,1), None, 
                    self.queue, (1,1,1), None,
                    np.int32(len(self.G.rxs)), np.int32(iteration), 
                    rxcoords_cl.data, rxs_cl.data,
                    self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data, 
                    self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                )
                store_output_event.wait()
                # self.G.Ex_cl.map_to_host()
                # self.G.Ey_cl.map_to_host()
                # self.G.Ez_cl.map_to_host()
                # self.G.Hx_cl.map_to_host()
                # self.G.Hy_cl.map_to_host()
                # self.G.Hz_cl.map_to_host()
                # print("G.Ex_cl : ", self.G.Ex_cl.get())
            # store any snapshots

            # update magnetic field components 
            # print("Updating Magnetic Field")
            # (int(np.ceil(((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))/256)),1,1)
            kernel_field_event = kernel_field_prg.update_h(
                # self.queue, (64*4,1,1), None,
                self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                self.G.ID_cl.data, self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data, 
                self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
            )
            kernel_field_event.wait()
            # self.G.Ex_cl.map_to_host()
            # self.G.Ey_cl.map_to_host()
            # self.G.Ez_cl.map_to_host()
            # self.G.Hx_cl.map_to_host()
            # self.G.Hy_cl.map_to_host()
            # self.G.Hz_cl.map_to_host()

            for pml in self.G.pmls:
                warnings.warn("Not implemented as of now")
                pass

            # update magnetic dipoles (sources)
            if self.G.magneticdipoles:
                raise NotImplementedError

            if Material.maxpoles != 0:
                raise NotImplementedError
            else:
                # print("Updating Electric Field")
                # update electric field components
                kernel_field_event = kernel_field_prg.update_e(
                    # self.queue, (64*4,1,1), (64,1,1),
                    self.queue, (2*int(np.ceil((self.G.nx+1)*(self.G.ny+1)*(self.G.nz+1))),1,1), None,
                    np.int32(self.G.nx), np.int32(self.G.ny), np.int32(self.G.nz),
                    self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data,
                    self.G.Hx_cl.data, self.G.Hy_cl.data, self.G.Hz_cl.data
                )
                kernel_field_event.wait()
                # self.G.Ex_cl.map_to_host()
                # self.G.Ey_cl.map_to_host()
                # self.G.Ez_cl.map_to_host()
                # self.G.Hx_cl.map_to_host()
                # self.G.Hy_cl.map_to_host()
                # self.G.Hz_cl.map_to_host()

            for pml in self.G.pmls:
                warnings.warn("Not implemented as of now")
                pass
            
            if self.G.voltagesources:
                raise NotImplementedError

            if self.G.hertziandipoles:
                # print("Updating Sources")
                # source_prg.printcoeffs(self.queue, (1,1,1), None)
                # dummy_src = np.empty_like(srcwaves_hertzian_cl, dtype=np.float32)*10
                # dummy_src_cl = cl_array.to_device(self.queue, dummy_src)
                # print("Src waves", srcwaves_hertzian_cl)
                source_event = source_prg.update_hertzian_dipole(
                    # self.queue, (round32(len(self.G.hertziandipoles)),1,1), None,
                    self.queue, (1,1,1), None,
                    np.int32(len(G.hertziandipoles)), np.int32(iteration),
                    np.float32(self.G.dx), np.float32(self.G.dy), np.float32(self.G.dz),
                    srcinfo1_hertzian_cl.data, srcinfo2_hertzian_cl.data, srcwaves_hertzian_cl.data, 
                    self.G.ID_cl.data, self.G.Ex_cl.data, self.G.Ey_cl.data, self.G.Ez_cl.data
                )
                source_event.wait()
                # self.G.Ex_cl.map_to_host()
                # self.G.Ey_cl.map_to_host()
                # self.G.Ez_cl.map_to_host()
                # self.G.Hx_cl.map_to_host()
                # self.G.Hy_cl.map_to_host()
                # self.G.Hz_cl.map_to_host()

                
            if Material.maxpoles > 0:
                raise NotImplementedError

        if self.G.rxs:
            # store the output from receivers array back to correct receiver objects
            gpu_get_rx_array(rxs_cl.get(), rxcoords_cl.get(), self.G)
        # print(rxs_cl.get())
        # print(rxcoords_cl.get())
        # copy data from any snapshots back to correct snapshot objects
        if self.G.snapshots:
            raise NotImplementedError

        # close context and queues
        return 1,2