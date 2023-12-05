.. image:: https://readthedocs.org/projects/gprmax/badge/?version=latest
    :target: http://docs.gprmax.com/en/latest/?badge=latest
    :alt: Documentation Status

|

.. image:: docs/source/images/gprMax_logo_small.png
    :target: http://www.gprmax.com

***************
Getting Started
***************

What is gprMax?
===============

`gprMax <http://www.gprmax.com>`_ is an open source software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is currently released under the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.

gprMax is principally written in `Python <https://www.python.org>`_ 3 with performance-critical parts written in `Cython <http://cython.org>`_. It includes a CPU-based solver parallelized using `OpenMP <http://www.openmp.org>`_, and a GPU-based solver written using the `NVIDIA CUDA <https://developer.nvidia.com/cuda-zone>`_ programming model.

Using gprMax? Cite us
---------------------

If you use gprMax and publish your work we would be grateful if you could cite our work using the following:

* Warren, C., Giannopoulos, A., & Giannakis I. (2016). gprMax: Open source software to simulate electromagnetic wave propagation for Ground Penetrating Radar, `Computer Physics Communications` (http://dx.doi.org/10.1016/j.cpc.2016.08.020)

For further information on referencing gprMax visit the `Publications section of our website <http://www.gprmax.com/publications.shtml>`_.


Package Overview
================

.. code-block:: bash

    gprMax/
        conda_env.yml
        CONTRIBUTORS
        docs/
        gprMax/
        gsoc/
        LICENSE
        README.rst
        setup.cfg
        setup.py
        tests/
        tools/
        user_libs/
        user_models/


* ``conda_env.yml`` is a configuration file for Anaconda (Miniconda) that sets up a Python environment with all the required Python packages for gprMax.
* ``CONTRIBUTORS`` contains a list of names of people who have contributed to the gprMax codebase.
* ``docs`` contains source files for the User Guide. The User Guide is written using `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ markup and is built using `Sphinx <http://sphinx-doc.org>`_ and `Read the Docs <https://readthedocs.org>`_.
* ``gprMax`` is the main package. Within this package, the main module is ``gprMax.py``
* ``gsoc`` contains information for `Google Summer of Code <https://summerofcode.withgoogle.com>`_ program - project ideas and proposal guidance.
* ``LICENSE`` contains information on the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``setup.cfg`` is used to set preferences for code formatting/styling using flake8.
* ``setup.py`` is used to compile the Cython extension modules.
* ``tests`` is a sub-package that contains test modules and input files.
* ``tools`` is a sub-package that contains scripts to assist with viewing and post-processing output from models.
* ``user_libs`` is a sub-package where useful modules contributed by users are stored.
* ``user_models`` is a sub-package where useful input files contributed by users are stored.

Installation
============

The following steps provide guidance on how to install gprMax:

1. Install Python, and the required Python packages, and get the gprMax source code from GitHub
2. Install a C compiler that supports OpenMP
3. Build and install gprMax

You can `watch screencasts <http://docs.gprmax.com/en/latest/screencasts.html>`_ that demonstrate the installation and update processes.

1. Install Python, the required Python packages, and get gprMax source
------------------------------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis.

* `Download and install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. Choose the Python 3.x version for your platform. We recommend choosing the installation options to: install Miniconda only for your user account; add Miniconda to your PATH environment variable; and register Miniconda Python as your default Python. See the `Quick Install page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for help installing Miniconda.
* Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:

.. code-block:: bash

    $ conda update conda
    $ conda install git
    $ git clone https://github.com/gprMax/gprMax.git
    $ cd gprMax
    $ conda env create -f conda_env.yml

This will ensure conda is up-to-date, install Git, get the latest gprMax source code from GitHub, and create an environment for gprMax with all the necessary Python packages.

If you prefer to install Python and the required Python packages manually, i.e. without using Anaconda/Miniconda, look in the ``conda_env.yml`` file for a list of the requirements.

If you are using Arch Linux (https://www.archlinux.org/) you may need to also install ``wxPython`` by adding it to the conda environment file (``conda_env.yml``).

2. Install a C compiler that supports OpenMP
---------------------------------------------

Linux
^^^^^

* `gcc <https://gcc.gnu.org>`_ should be already installed, so no action is required.


macOS
^^^^^

* Xcode (the IDE for macOS) comes with the LLVM (clang) compiler, but it does not currently support OpenMP, so you must install `gcc <https://gcc.gnu.org>`_. That said, it is still useful to have Xcode (with command line tools) installed. It can be downloaded from the App Store. Once Xcode is installed, download and install the `Homebrew package manager <http://brew.sh>`_ and then to install gcc, run:

.. code-block:: bash

    $ brew install gcc

Microsoft Windows
^^^^^^^^^^^^^^^^^
 
* Download and install Microsoft `Build Tools for Visual Studio 2022 <https://aka.ms/vs/17/release/vs_BuildTools.exe>`_ (direct link). You can also find it on the `Microsoft Visual Studio downloads page <https://visualstudio.microsoft.com/downloads/>`_ by scrolling down to the 'All Downloads' section, clicking the disclosure triangle by 'Tools for Visual Studio 2022', then clicking the download button next to 'Build Tools for Visual Studio 2022'. When installing, choose the 'Desktop development with C++' Workload and select only 'MSVC v143' and 'Windows 10 SDK' or 'Windows 11 SDK options.
* Set the Path and Environment Variables - this can be done by following the `instructions from Microsoft <https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_file_locations>`_, or manually by adding a form of :code:`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64` (this may vary according to your exact machine and installation) to your system Path environment variable.

Alternatively, if you are using Windows 10/11 you can install the `Windows Subsystem for Linux <https://docs.microsoft.com/en-gb/windows/wsl/about>`_ and then follow the Linux install instructions for gprMax. Note however that currently WSL does not aim to support GUI desktops or applications, e.g. Gnome, KDE, etc....

3. Build and install gprMax
---------------------------

Once you have installed the aforementioned tools follow these steps to build and install gprMax:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment:code:`conda activate gprMax`. Run the following commands:

.. code-block:: bash

    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

**You are now ready to proceed to running gprMax.**

Running gprMax
==============

gprMax is designed as a Python package, i.e. a namespace that can contain multiple packages and modules, much like a directory.

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment:code:`conda activate gprMax`.

Basic usage of gprMax is:

.. code-block:: bash

    (gprMax)$ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: bash

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: bash

    (gprMax)$ python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out

Your results should be like those from the A-scan from the metal cylinder example in `introductory/basic 2D models section <http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results>`_

When you are finished using gprMax, the conda environment can be deactivated using :code:`conda deactivate`.

Optional command line arguments
-------------------------------

====================== ========= ===========
Argument name          Type      Description
====================== ========= ===========
``-n``                 integer   number of times to run the input file. This option can be used to run a series of models, e.g. to create a B-scan with 60 traces: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60``
``-gpu``               flag/list flag to use NVIDIA GPU or list of NVIDIA GPU device ID(s) for specific GPU card(s), e.g. ``-gpu 0 1``
``-restart``           integer   model number to start/restart the simulation from. It would typically be used to restart a series of models from a specific model number, with the ``-n`` argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 15 -restart 45``
``-task``              integer   task identifier (model number) when running the simulation as a job array on `Open Grid Scheduler/Grid Engine <http://gridscheduler.sourceforge.net/index.html>`_. For further details see the `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
``-mpi``               integer   number of Message Passing Interface (MPI) tasks, i.e. master + workers, for MPI task farm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI task farm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60 -mpi 61``. For further details see the `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
``--mpi-no-spawn``     flag      uses MPI task farm without spawn mechanism. For further details see the `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
``-benchmark``         flag      switch on benchmarking mode. This can be used to benchmark the threading (parallel) performance of gprMax on different hardware. For further details see the `benchmarking section of the User Guide <http://docs.gprmax.com/en/latest/benchmarking.html>`_
``--geometry-only``    flag      build a model and produce any geometry views but does not run the simulation, e.g. to check the geometry of a model is correct: ``(gprMax)$ python -m gprMax user_models/heterogeneous_soil.in --geometry-only``
``--geometry-fixed``   flag      runs a series of models where the geometry does not change between models, e.g. a B-scan where *only* the position of simple sources and receivers, moved using ``#src_steps`` and ``#rx_steps``, changes between models.
``--opt-taguchi``      flag      runs a series of models using an optimization process based on Taguchi's method. For further details see the `user libraries section of the User Guide <http://docs.gprmax.com/en/latest/user_libs_opt_taguchi.html>`_
``--write-processed``  flag      writes another input file after any Python code and include commands in the original input file have been processed. Useful for checking that any Python code is being correctly processed into gprMax commands.
``-h`` or ``--help``   flag      is used to get help on command line options.
====================== ========= ===========

Updating gprMax
===============

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run the following commands:

.. code-block:: bash

    (gprMax)$ git pull
    (gprMax)$ python setup.py cleanall
    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

This will pull the most recent gprMax source code from GitHub, remove/clean previously built modules, and then build and install the latest version of gprMax.


Updating conda and Python packages
----------------------------------

Periodically you should update conda and the required Python packages. With the gprMax environment deactivated and from the top-level gprMax directory, run the following commands:

.. code-block:: bash

    $ conda update conda
    $ conda env update -f conda_env.yml

Thanks To Our Contributors ✨🔗
==========================
.. image:: https://contrib.rocks/image?repo=gprMax/gprMax
   :target: https://github.com/gprMax/gprMax/graphs/contributors