.. _gpu:

**********
Using GPUs
**********

The most computationally intensive parts of gprMax, which are the FDTD solver loops, can optionally be executed using General-purpose computing on graphics processing units (GPGPU). This has been achieved through use of the NVIDIA CUDA programming environment, therefore a `NVIDIA CUDA-Enabled GPU <https://developer.nvidia.com/cuda-gpus>`_ is required to take advantage of the GPU-based solver.

Extra installation steps for GPU usage
======================================

The following steps provide guidance on how to install the extra components to allow gprMax to run on your GPU:

1. Install the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_. You can follow the Installation Guides in the `NVIDIA CUDA Toolkit Documentation <http://docs.nvidia.com/cuda/index.html#installation-guides>`_
2. Install the pycuda Python module. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/macOS) or :code:`activate gprMax` (Windows). Run :code:`conda install pycuda`


Run a test model
================

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/macOS) or :code:`activate gprMax` (Windows)

Run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in -gpu
