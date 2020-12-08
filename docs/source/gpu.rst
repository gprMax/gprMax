.. _gpu:

*****
GPGPU
*****

The most computationally intensive parts of gprMax, which are the FDTD solver loops, can optionally be executed using General-purpose computing on graphics processing units (GPGPU). This has been achieved through use of the NVIDIA CUDA programming environment, therefore a `NVIDIA CUDA-Enabled GPU <https://developer.nvidia.com/cuda-gpus>`_ is required to take advantage of the GPU-based solver.

Extra installation steps for GPU usage
======================================

The following steps provide guidance on how to install the extra components to allow gprMax to run on your NVIDIA GPU:

1. Install the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_. You can follow the Installation Guides in the `NVIDIA CUDA Toolkit Documentation <http://docs.nvidia.com/cuda/index.html#installation-guides>`_ You must ensure the version of CUDA you install is compatible with the compiler you are using. This information can usually be found in a table in the CUDA Installation Guide under System Requirements.
2. You may need to add the location of the CUDA compiler (:code:`nvcc`) to your user path environment variable, e.g. for Windows :code:`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin` or Linux/macOS :code:`/Developer/NVIDIA/CUDA-X.X/bin`.
3. Install the pycuda Python module. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run :code:`pip install pycuda`

Running gprMax using GPU(s)
===========================

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`

Run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in -gpu

.. note::

    If you want to select a specific GPU card on your system, you can specify an integer after the :code:`-gpu` flag. The integer should be the NVIDIA CUDA device ID for a specific GPU card. If it is not specified it defaults to device ID 0.


Combining MPI and GPU usage
---------------------------

Message Passing Interface (MPI) has been utilised to implement a simple task farm that can be used to distribute a series of models as independent tasks. This is described in more detail in the :ref:`OpenMP, MPI, HPC section <openmp-mpi>`. MPI can be combined with the GPU functionality to allow a series models to be distributed to multiple GPUs on the same machine (node). For example, to run a B-scan that contains 60 A-scans (traces) on a system with 4 GPUs:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60 -mpi 5 -gpu 0 1 2 3

.. note::

    The argument given with `-mpi` is number of MPI tasks, i.e. master + workers, for MPI task farm. So in this case, 1 master (CPU) and 4 workers (GPU cards). The integers given with the `-gpu` argument are the NVIDIA CUDA device IDs for the specific GPU cards to be used.
