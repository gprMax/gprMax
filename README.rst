.. image:: https://readthedocs.org/projects/gprmax/badge/?version=devel
    :target: http://docs.gprmax.com/en/latest/?badge=devel
    :alt: Documentation Status

|

.. image:: images_shared/gprMax_logo_small.png
    :target: http://www.gprmax.com
    :alt: gprMax

.. include_in_docs_after_this_label

***************
Getting Started
***************

What is gprMax?
===============

`gprMax <http://www.gprmax.com>`_ is an open source software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is currently released under the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.

gprMax is principally written in `Python <https://www.python.org>`_ 3 with performance-critical parts written in `Cython <http://cython.org>`_. It includes accelerators for CPU using `OpenMP <http://www.openmp.org>`_, CPU/GPU using `OpenCL <https://www.khronos.org/api/opencl>`_, and GPU using `NVIDIA CUDA <https://developer.nvidia.com/cuda-zone>`_. Additionally, MPI support (using `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_) enables larger scale (multi-node) simulations. There is more information about the different acceleration approaches in the :ref:`accelerators` section.

Using gprMax? Cite us
---------------------

If you use gprMax and publish your work we would be grateful if you could cite our work using:

* Warren, C., Giannopoulos, A., & Giannakis I. (2016). gprMax: Open source software to simulate electromagnetic wave propagation for Ground Penetrating Radar, `Computer Physics Communications` (http://dx.doi.org/10.1016/j.cpc.2016.08.020)

For further information on referencing gprMax visit the `Publications section of our website <http://www.gprmax.com/publications.shtml>`_.


Package overview
================

.. code-block:: bash

    gprMax/
        docs/
        examples/
        gprMax/
        reframe_tests/
        testing/
        toolboxes/
        CITATION.cff
        CODE_OF_CONDUCT.md
        conda_env.yml
        CONTRIBUTING.md
        CREDITS
        LICENSE
        MANIFEST.in
        pyproject.toml
        README.rst
        requirements.txt
        setup.py

* ``docs/`` contains source files for the User Guide. The User Guide is written using `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ markup, and is built using `Sphinx <http://sphinx-doc.org>`_ and `Read the Docs <https://readthedocs.org>`_.
* ``examples/`` is a sub-package where example input files and models are stored.
* ``gprMax/`` is the main package. Within this package, the main module is ``gprMax.py``
* ``reframe_tests/`` contains regression tests run using
  `ReFrame <https://reframe-hpc.readthedocs.io>`_. The regression checks are currently specific to the `ARCHER2 <https://www.archer2.ac.uk/>`_ system and additional work wil be required to make them portable between systems.
* ``testing/`` is a sub-package which contains test modules and input files.
* ``toolboxes/`` is a sub-package where useful modules contributed by users are stored.
* ``CITATION.cff`` is a plain text file with human- and machine-readable citation information for gprMax.
* ``conda_env.yml`` is a configuration file for Anaconda (Miniconda) that sets up a Python environment with all the required Python packages for gprMax.
* ``CONTRIBUTING.md`` is guide on how to contribute to gprMax.
* ``CREDITS`` contains a list of names of people who have contributed to the gprMax codebase.
* ``LICENSE`` contains information on the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.
* ``MANIFEST.in`` consists of commands, one per line, instructing setuptools to add or remove files from the source distribution.
* ``pyproject.toml`` contains build system requirements.
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``requirements.txt`` is a configuration file for pip that sets up a Python environment with all the required Python packages for gprMax.
* ``setup.py`` is the centre of all activity in building, distributing, and installing gprMax, including building and compiling the Cython extension modules.



Installation
============

The following steps provide guidance on how to install gprMax:

1. Install a C compiler which supports OpenMP
2. Install MPI
3. Install FFTW
4. Install Python, required Python packages, and get the gprMax source code from GitHub
5. [Optional] Build h5py against Parallel HDF5
6. Build and install gprMax

1. Install a C compiler which supports OpenMP
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
* Set the Path and Environment Variables - this can be done by following the `instructions from Microsoft <https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_file_locations>`_, or manually by adding a form of ``C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64`` (this may vary according to your exact machine and installation) to your system Path environment variable.

Alternatively, if you are using Windows 10/11 you can install the `Windows Subsystem for Linux <https://docs.microsoft.com/en-gb/windows/wsl/about>`_ and then follow the Linux install instructions for gprMax. Note however that currently, WSL does not aim to support GUI desktops or applications, e.g. Gnome, KDE, etc...


2. Install MPI
--------------

If you are running gprMax on a HPC system, MPI will likely be installed already. Otherwise you will need to install it yourself.

Linux/macOS
^^^^^^^^^^^
* It is recommended to use `OpenMPI <http://www.open-mpi.org>`_.

Microsoft Windows
^^^^^^^^^^^^^^^^^
* It is recommended to use `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_. Download and install both the .exe and .msi files.


3. Install FFTW
---------------
If you are running gprMax on a HPC system, FFTW may be available already - consult your site's documentation. Otherwise you will need to install it yourself.

Linux
^^^^^
* It is possible binaries are available via your package manager. E.g. ``libfftw3-dev`` on Ubuntu.
* Otherwise you can find the latest source code on the `fftw downloads page <https://fftw.org/download.html>`_. There are instructions to build from source in the `fftw docs <https://fftw.org/fftw3_doc/Installation-on-Unix.html>`_.

macOS
^^^^^
* FFTW can be installed using the `Homebrew package manager <http://brew.sh>`_:

.. code-block:: bash

    $ brew install fftw

Microsoft Windows
^^^^^^^^^^^^^^^^^
* There is guidance to install FFTW on Windows available `here <https://fftw.org/install/windows.html>`_.


4. Install Python, the required Python packages, and get the gprMax source
--------------------------------------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis.

* `Download and install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. Choose the Python 3.x version for your platform. We recommend choosing the installation options to: install Miniconda only for your user account; add Miniconda to your PATH environment variable; and register Miniconda Python as your default Python. See the `Quick Install page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for help installing Miniconda.
* Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:

.. code-block:: bash

    $ conda update conda
    $ conda install git
    $ git clone https://github.com/gprMax/gprMax.git
    $ cd gprMax
    $ conda env create -f conda_env.yml

This will make sure conda is up-to-date, install Git, get the latest gprMax source code from GitHub, and create an environment for gprMax with all the necessary Python packages.

If you prefer to install Python and the required Python packages manually, i.e. without using Anaconda/Miniconda, look in the ``conda_env.yml`` file for a list of the requirements.

If you are using Arch Linux (https://www.archlinux.org/) you may need to also install ``wxPython`` by adding it to the conda environment file (``conda_env.yml``).


.. _h5py_mpi:

5. [Optional] Build h5py against Parallel HDF5
----------------------------------------------

If you plan to use the :ref:`MPI domain decomposition functionality <mpi_domain_decomposition>` available in gprMax, h5py must be built with MPI support.

Install with conda
^^^^^^^^^^^^^^^^^^

h5py can be installed with MPI support in a conda environment with:

.. code:: bash

    conda install "h5py>=2.9=mpi*"

Install with pip
^^^^^^^^^^^^^^^^

Set your default compiler to the ``mpicc`` wrapper and build h5py with the ``HDF5_MPI`` environment variable:

.. code:: bash

    export CC=mpicc
    export HDF5_MPI="ON"
    pip install --no-binary=h5py h5py  # Add --no-cache-dir if pip has cached a previous build of h5py

Further guidance on building h5py against a parallel build of HDF5 is available in the `h5py documentation <https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5>`_.


6. Build and install gprMax
---------------------------

Once you have installed the aforementioned tools follow these steps to build and install gprMax:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), **navigate into the directory above the gprMax package**, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run the following commands:

.. code-block:: bash

    (gprMax)$ pip install -e gprMax

**You are now ready to proceed to running gprMax.**

Running gprMax
==============

gprMax is designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory.

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`.

Basic usage of gprMax is:

.. code-block:: bash

    (gprMax)$ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: bash

    (gprMax)$ python -m gprMax examples/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: bash

    (gprMax)$ python -m toolboxes.Plotting.plot_Ascan examples/cylinder_Ascan_2D.h5

Your results should be like those from the A-scan from the metal cylinder example in `introductory/basic 2D models section <http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results>`_

When you are finished using gprMax, the conda environment can be deactivated using :code:`conda deactivate`.

Optional command line arguments
-------------------------------

..  list-table::
    :widths: 40 10 50
    :header-rows: 1

    * - Argument name
      - Type
      - Description
    * - ``-o`` or ``-outputfile``
      - string
      - File path to save the output data.
    * - ``-n``
      - integer
      - Number of required simulation runs. This option can be used to run a series of models, e.g. to create a B-scan with 60 traces: ``(gprMax)$ python -m gprMax examples/cylinder_Bscan_2D.in -n 60``
    * - ``-i``
      - integer
      - Model number to start/restart the simulation from. It would typically be used to restart a series of models from a specific model number, with the n argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces.
    * - ``-t`` or ``--taskfarm``
      - flag
      - Flag to use Message Passing Interface (MPI) taskfarm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI taskfarm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``(gprMax)$ python -m gprMax examples/cylinder_Bscan_2D.in -n 60 --taskfarm``. For further details see the
        `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
    * - ``--mpi``
      - list
      - Flag to use Message Passing Interface (MPI) to divide the model between MPI ranks. Three integers should be provided to define the number of MPI processes (min 1) in the x, y, and z dimensions.
    * - ``-gpu``
      - list/bool
      - Flag to use NVIDIA GPU or list of NVIDIA GPU device ID(s) for specific GPU card(s), e.g. ``-gpu 0 1``
    * - ``-opencl``
      - list/bool
      - Flag to use OpenCL or list of OpenCL device ID(s) for specific compute device(s).
    * - ``--geometry-only``
      - flag
      - Build a model and produce any geometry views but do not run the simulation, e.g. to check
        the geometry of a model is correct: ``(gprMax)$ python -m gprMax examples/heterogeneous_soil.in --geometry-only``
    * - ``--geometry-fixed``
      - flag
      - Run a series of models where the geometry does not change between models, e.g. a B-scan where *only* the position of simple sources and receivers, moved using ``#src_steps`` and ``#rx_steps``, changes between models.
    * - ``--write-processed``
      - flag
      - Write another input file after any Python blocks and include commands in the original input file have been processed. Useful for checking that any Python blocks are being correctly processed into gprMax commands.
    * - ``--show-progress-bars``
      - flag
      - Forces progress bars to be displayed - by default, progress bars are displayed when the log level is info (20) or less.
    * - ``--hide-progress-bars``
      - flag
      - Forces progress bars to be displayed - by default, progress bars are hidden when the log level is greater than info (20).
    * - ``--log-level``
      - integer
      - Level of logging to use, see the `Python logging module <https://docs.python.org/3/library/logging.html>`_.
    * - ``--log-file``
      - flag
      - Write logging information to file.
    * - ``--log-all-ranks``
      - flag
      - Write logging information from all MPI ranks. Default behaviour only provides log output
        from rank 0. When used with --log-file, each rank will write to an individual file.
    * - ``-h`` or ``--help``
      - flag
      - Used to get help on command line options.

Updating gprMax
===============

* The safest and simplest way to upgrade gprMax is to uninstall, clone the latest version, and re-install the software. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the directory above the gprMax package, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run the following command:

.. code-block:: bash

    (gprMax)$ pip uninstall gprMax
    (gprMax)$ git clone https://github.com/gprMax/gprMax.git
    (gprMax)$ pip install -e gprMax

This will uninstall gprMax, clone the most recent gprMax source code from GitHub, and then build and install the latest version of gprMax.


Updating conda and Python packages
----------------------------------

Periodically you should update conda and the required Python packages. With the gprMax environment deactivated and from the top-level gprMax directory, run the following commands:

.. code-block:: bash

    $ conda update conda
    $ conda env update -f conda_env.yml


Thanks To Our Contributors ✨🔗
===============================
.. image:: https://contrib.rocks/image?repo=gprMax/gprMax
   :target: https://github.com/gprMax/gprMax/graphs/contributors
