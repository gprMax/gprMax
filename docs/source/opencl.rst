*************
OpenCL Solver
*************

gprMax now supports simulations accelerated with the help of OpenCl. OpenCl is a framework for writing programs that execute across heterogeneous platforms consisting of CPUs, GPUs or even FPGAs. Due to removal of dependency on the type of computing device that will be used, OpenClSolver can run on possibly any device which supports OpenCl. 

Installation steps for OpenCl Usage
===================================

The following steps provide essentials steps for required to run gprMax with OpenCl enabled. 

1. Install OpenCL ICD required to run general OpenCl Programs. For Intel Processors (CPU/GPU) download the OpenCL SDK from `Intel official page <https://software.intel.com/en-us/opencl-sdk>`_. For AMD download and install the AMD OpenCL SDK from `Windows <https://www.softpedia.com/get/Programming/SDK-DDK/ATI-Stream-SDK.shtml>`_ / `Linux <https://sourceforge.net/projects/nicehashsgminerv5viptools/files/APP%20SDK%20A%20Complete%20Development%20Platform/>`_ and similarly for NVIDIA GPU install `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_

2. Install PyOpenCl using conda in your ``gprMax`` conda environment using ``conda install -c conda-forge pyopencl``.

3. You may have to add path of OpenCl ``Cl/cl.h`` to your environment variables.

Running OpenCl Solver
=====================

Open a Terminal or Command Prompt and navigate to top-level gprMax directory. Activate the gprMax conda environment if not already activated with ``conda activate gprMax``.

Run OpenCl under standard conditions with:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in --opencl

Using OpenCl with a GPU Device only:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in --opencl -gpu

Using OpenCl with special ElementwiseKernel tool:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in --opencl --elwise

.. note::

    For every run on OpenCl Solver, user are prompt for the choice of Platform and Device. In case gpu is toggled, only GPU device are shown. 