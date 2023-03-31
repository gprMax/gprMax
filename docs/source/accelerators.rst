.. _accelerators:

******************
OpenMP/CUDA/OpenCL
******************

The most computationally intensive parts of gprMax, which are the FDTD solver loops, have been parallelised using different CPU and GPU accelerators to offer performance and flexibility.

1. `OpenMP <http://openmp.org>`_ which supports multi-platform shared memory multiprocessing.
2. `NVIDIA CUDA <https://developer.nvidia.com/cuda-toolkit>`_ for NVIDIA GPUs.
3. `OpenCL <https://www.khronos.org/api/opencl>`_ for a wider range of CPU and GPU hardware. 

Additionally the Message Passing Interface (MPI) can be utilised to implement a simple task farm that can be used to distribute a series of models as independent tasks. This can be useful in many GPR simulations where a B-scan (composed of multiple A-scans) is required. Each A-scan can be task-farmed as a independent model, and within each model OpenMP or CUDA can still be used for parallelism. This creates mixed mode OpenMP/MPI or CUDA/MPI environments.

Some of these accelerators and frameworks require additional software to be installed. The guidance below explains how to do that and gives examples of usage.

.. note::

    You can use the ``get_host_spec.py`` module (in ``toolboxes/Utilities``) to help you understand what hardware (CPU/GPU) you have and how gprMax can use it with the aforementioned accelerators.


OpenMP
======

No additional software is required to use OpenMP as it is part of the standard installation of gprMax.

By default gprMax will try to determine and use the maximum number of OpenMP threads (usually the number of physical CPU cores) available on your machine. You can override this behaviour in two ways: firstly, gprMax will check to see if the ``#cpu_threads`` command is present in your input file; if not, gprMax will check to see if the environment variable ``OMP_NUM_THREADS`` is set. This can be useful if you are running gprMax in a High-Performance Computing (HPC) environment where you might not want to use all of the available CPU cores.

MPI
===

By default the MPI task farm functionality is turned off. It can be used with the ``-mpi`` command line option, which specifies the total number of MPI tasks, i.e. master + workers, for the MPI task farm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI task farm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``(gprMax)$ python -m gprMax examples/cylinder_Bscan_2D.in -n 60 -mpi 61``.

Software required
-----------------

The following steps provide guidance on how to install the extra components to allow the MPI task farm functionality with gprMax:

1. Install MPI on your system.

Linux/macOS
^^^^^^^^^^^
It is recommended to use `OpenMPI <http://www.open-mpi.org>`_.

Microsoft Windows
^^^^^^^^^^^^^^^^^
It is recommended to use `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_. Download and install both the .exe and .msi files.

2. Install the ``mpi4py`` Python module. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run :code:`pip install mpi4py`


CUDA
====

Software required
-----------------

The following steps provide guidance on how to install the extra components to allow gprMax to run on your NVIDIA GPU:

1. Install the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_. You can follow the Installation Guides in the `NVIDIA CUDA Toolkit Documentation <http://docs.nvidia.com/cuda/index.html#installation-guides>`_ You must ensure the version of CUDA you install is compatible with the compiler you are using. This information can usually be found in a table in the CUDA Installation Guide under System Requirements.
2. You may need to add the location of the CUDA compiler (``nvcc``) to your user path environment variable, e.g. for Windows ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin`` or Linux/macOS ``/Developer/NVIDIA/CUDA-X.X/bin``.
3. Install the pycuda Python module. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment ``conda activate gprMax``. Run ``pip install pycuda``

Example
-------

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment ``conda activate gprMax``

Run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax examples/cylinder_Ascan_2D.in -gpu

.. note::

    * If you want to select a specific GPU card on your system, you can specify an integer after the ``-gpu`` flag. The integer should be the NVIDIA CUDA device ID for a specific GPU card. If it is not specified it defaults to device ID 0. 
    * You can use the ``get_host_spec.py`` module (in ``toolboxes/Utilities``) to help you understand what hardware (CPU/GPU) you have and how gprMax can use it.


OpenCL
======

Software required
-----------------

The following steps provide guidance on how to install the extra components to allow gprMax to use OpenCL:

1. Install the pyopencl Python module. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment ``conda activate gprMax``. Run ``pip install pyopencl``

Example
-------

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment ``conda activate gprMax``

Run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax examples/cylinder_Ascan_2D.in -opencl

.. note::

    * If you want to select a specific computer device on your system, you can specify an integer after the ``-opencl`` flag. The integer should be the device ID for a specific compute device. If it is not specified it defaults to device ID 0.
    * You can use the ``get_host_spec.py`` module (in ``toolboxes/Utilities``) to help you understand what hardware (CPU/GPU) you have and how gprMax can use it.


CUDA/MPI
========

Message Passing Interface (MPI) has been utilised to implement a simple task farm that can be used to distribute a series of models as independent tasks. This is described in more detail in the :ref:`HPC <hpc>` section. MPI can be combined with the GPU functionality to allow a series models to be distributed to multiple GPUs on the same machine (node). 

Example
-------

For example, to run a B-scan that contains 60 A-scans (traces) on a system with 4 GPUs:

.. code-block:: none

    (gprMax)$ python -m gprMax examples/cylinder_Bscan_2D.in -n 60 -mpi 5 -gpu 0 1 2 3

.. note::

    The argument given with ``-mpi`` is number of MPI tasks, i.e. master + workers, for MPI task farm. So in this case, 1 master (CPU) and 4 workers (GPU cards). The integers given with the ``-gpu`` argument are the NVIDIA CUDA device IDs for the specific GPU cards to be used.
