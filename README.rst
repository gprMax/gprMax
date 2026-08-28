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

`gprMax <http://www.gprmax.com>`_ is open-source computational
electromagnetics software that solves Maxwell's equations using the
finite-difference time-domain (FDTD) method. It supports two- and
three-dimensional models through both a Python API and a text-based input-file
interface.

The software originated in research on the forward problem of ground-penetrating
radar in the 1990s. Antonis Giannopoulos created the original gprMax code and
established its numerical foundations; this work is documented in his `1997
D.Phil. thesis <https://etheses.whiterose.ac.uk/id/eprint/2443>`_. Ground
Penetrating Radar (GPR) remains an important application and gives gprMax its
name.

Craig Warren led the creation of the open-source Python and Cython codebase
that became gprMax version 3, building on `his doctoral research
<https://era.ed.ac.uk/items/0fd15c09-cb97-47a0-98b4-9ae0b13e81a2>`_. Iraklis
Giannakis contributed major original developments in dispersive-material
modelling, fractal media and realistic GPR models through `his doctoral
research on realistic GPR modelling
<https://era.ed.ac.uk/items/1e853a06-d597-4bdf-964c-82cdda258683>`_.
Together with Antonis Giannopoulos, they described this generation in the
`2016 Computer Physics Communications paper
<https://doi.org/10.1016/j.cpc.2016.08.020>`_.

Version 4 was initiated through the `doctoral research of John Hartley
<https://era.ed.ac.uk/items/d253612b-7c1e-4adf-9a39-d730bbe76a95>`_, including
the development of FDTD subgridding and dispersive-interface averaging. Nathan
Mannall subsequently carried out a major architectural refactoring and
developed the MPI domain-decomposition solver. Qifeng Shen developed the FDFD
eigenmode solver, wave-port and impedance-boundary capabilities, and has led
the expansion of antenna and RF modelling. Accelerator backends, testing
infrastructure, geometry importers, toolboxes, antenna models and documentation
also reflect substantial work by the wider gprMax community and Google Summer
of Code contributors. The `full authors and contributors record
<https://github.com/gprMax/gprMax/blob/devel/AUTHORS.rst>`_ describes these
contributions in more detail.

After nearly three decades of development, the current version 4 codebase
extends substantially beyond the capabilities described in the 2016 paper.
Although GPR remains a core application, gprMax is now a general-purpose
research platform for time-domain computational electromagnetics. Applications
also include antenna and microwave modelling, electromagnetic scattering and
radar cross section, bioelectromagnetics and dosimetry, and radiometry. A range
of established and recent methods from the published literature complement
gprMax's original formulations and existing core functionality. Some of the
key features include:

* geometrical modelling with dielectric smoothing, semantic object tags,
  imported voxel geometries, fractal media and locally refined subgrids;
* conductive, anisotropic, magnetic and multipole dispersive materials, with
  reusable material databases and dispersive interface averaging;
* dipole, voltage, transmission-line, magnetic-frill, plane-wave, rational
  network and FDFD eigenmode-port excitation;
* port quantities and multi-case studies, S-parameters, impedance, antenna
  gain and directivity, radar cross section, near- and far-field transforms,
  SAR and radiometric absorbed-power outputs; and
* shared-memory CPU execution, NVIDIA CUDA, OpenCL, Apple Metal and MPI domain
  decomposition, with common model-building and output interfaces.

gprMax is currently released under the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.

gprMax is principally written in `Python <https://www.python.org>`_ 3 with performance-critical parts written in `Cython <http://cython.org>`_. It includes accelerators for CPU using `OpenMP <http://www.openmp.org>`_, CPU/GPU using `OpenCL <https://www.khronos.org/api/opencl>`_, GPU using `NVIDIA CUDA <https://developer.nvidia.com/cuda-zone>`_, and GPU using `Apple Metal <https://developer.apple.com/metal/>`_ on macOS with M-series chips. Additionally, MPI support (using `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_) enables larger scale (multi-node) simulations. There is more information about the different acceleration approaches in the performance section of the documentation.

Using gprMax? Cite us
---------------------

If you use gprMax and publish your work we would be grateful if you could cite our work using:

* Warren, C., Giannopoulos, A., & Giannakis I. (2016). gprMax: Open source software to simulate electromagnetic wave propagation for Ground Penetrating Radar, `Computer Physics Communications` (http://dx.doi.org/10.1016/j.cpc.2016.08.020)

For further information on referencing gprMax visit the `Publications section of our website <http://www.gprmax.com/publications.shtml>`_.


Repository overview
===================

.. code-block:: none

    gprMax/
        .github/
        docs/
        examples/
        gprMax/
        images_shared/
        packaging/
        reframe_tests/
        testing/
        tests/
        toolboxes/
        CITATION.cff
        CODE_OF_CONDUCT.md
        conda_env.yml
        CONTRIBUTING.md
        AUTHORS.rst
        LICENSE
        MANIFEST.in
        pyproject.toml
        README.rst
        requirements.txt
        build_config.py
        packaging_config.py
        setup.py

* ``.github/`` contains the continuous-integration workflows that test the
  supported operating systems, build binary wheels and source distributions,
  and run the automated test suites.
* ``docs/`` contains the source for the User Guide. It uses
  `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
  and `Sphinx <https://www.sphinx-doc.org>`_, and is published by
  `Read the Docs <https://readthedocs.org>`_.
* ``examples/`` contains user-facing input files and Python API models grouped
  by application. These examples are also distributed as package resources so
  they remain available to users who install a binary wheel.
* ``gprMax/`` is the main Python package. It contains model construction,
  material and source definitions, CPU and accelerator solvers, MPI domain
  decomposition, subgridding, FDFD eigenmode ports, near-to-far-field
  transforms, antenna and port processing, SAR and radiometry outputs, and the
  compiled Cython kernels used by performance-critical operations.
* ``images_shared/`` stores figures shared by the README and User Guide.
* ``packaging/`` contains platform-specific helpers used to produce portable
  binary distributions, including the macOS OpenMP build support.
* ``reframe_tests/`` contains whole-model and HPC regression tests built with
  `ReFrame <https://reframe-hpc.readthedocs.io>`_. The supplied machine
  configuration and numerical references currently target
  `ARCHER2 <https://www.archer2.ac.uk/>`_; other systems can provide their own
  configuration and reference data.
* ``testing/`` is the manually run scientific evidence archive. It separates
  analytical validation, comparisons with other numerical codes, backend
  consistency studies, larger regression campaigns, experimental work, and
  performance benchmarks.
* ``tests/`` is the automated pytest suite, containing focused unit tests,
  compact integration tests, platform tests and tests that require real
  accelerator hardware.
* ``toolboxes/`` contains user-facing processing, conversion, visualisation,
  antenna-model and waveform-modelling tools. Toolboxes and their compact
  examples are included in source and binary distributions.
* ``CITATION.cff`` is a plain text file with human- and machine-readable citation information for gprMax.
* ``conda_env.yml`` defines the recommended Conda development environment.
  MPI runtimes and accelerator bindings remain optional and are installed for
  the hardware and workflow being used.
* ``CONTRIBUTING.md`` is a guide to contributing to gprMax.
* ``AUTHORS.rst`` records the people and organisations that have created,
  developed, contributed to, and supported gprMax.
* ``LICENSE`` contains information on the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.
* ``MANIFEST.in`` consists of commands, one per line, instructing setuptools to add or remove files from the source distribution.
* ``pyproject.toml`` contains build-system requirements and configuration for
  pytest, source formatting, and cross-platform binary-wheel builds.
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``requirements.txt`` lists the common source-development and test
  dependencies that can be installed with pip. Optional MPI and accelerator
  dependencies are selected through package extras.
* ``build_config.py`` provides the portable compiler and OpenMP configuration
  used for local source builds and binary wheels.
* ``packaging_config.py`` defines which packages, examples, toolboxes and data
  files are included in installed distributions.
* ``setup.py`` defines the setuptools package metadata and Cython extension
  modules, using the shared build and packaging configuration above.

.. _installation:

Installation
============

Binary wheel installation (recommended)
---------------------------------------

For a released gprMax v4 package, the simplest installation is:

.. code-block:: console

    $ python -m pip install gprMax

The binary wheels contain the compiled CPU extensions and do not require a C
compiler, OpenMP development headers, Git, or a Conda environment on the
user's machine. Python 3.11--3.13 is supported, with Python 3.12 recommended.
Release wheels are built for 64-bit Linux and Windows, and for both Intel and
Apple Silicon macOS. A virtual environment, including one created by Conda or
``venv``, is still recommended to isolate dependencies.

Optional features use the same extras as source installations, for example:

.. code-block:: console

    $ python -m pip install "gprMax[mpi]"
    $ python -m pip install "gprMax[cuda]"
    $ python -m pip install "gprMax[metal]"

These extras install Python bindings only. MPI, CUDA, OpenCL, and Metal still
require compatible system runtimes and hardware as described below and in
the `accelerator documentation
<https://docs.gprmax.com/en/latest/accelerators.html>`_.

Installing from source
----------------------

Build from source when developing gprMax, modifying its Cython extensions, or
using a platform for which no wheel is published. The source-build steps are:

1. Install a C compiler which supports OpenMP
2. [Optional] Install MPI
3. Install Python, required Python packages, and get the gprMax source code from GitHub
4. [Optional] Build h5py against Parallel HDF5
5. [Optional] Install mpi4py_fft
6. Build and install gprMax

1. Install a C compiler which supports OpenMP
---------------------------------------------

Linux
^^^^^

* `gcc <https://gcc.gnu.org>`_ should be already installed, so no action is required.


macOS
^^^^^

* Install the Xcode command-line tools, which provide Apple Clang, and install
  the OpenMP runtime using `Homebrew <https://brew.sh>`_:

.. code-block:: console

    $ xcode-select --install
    $ brew install libomp

  gprMax detects ``libomp`` from Homebrew, the active Python environment, or
  ``GPRMAX_LIBOMP_PREFIX``. The previous Homebrew GCC requirement is no longer
  necessary.

Microsoft Windows
^^^^^^^^^^^^^^^^^

* Download and install Microsoft `Build Tools for Visual Studio 2022 <https://aka.ms/vs/17/release/vs_BuildTools.exe>`_ (direct link). You can also find it on the `Microsoft Visual Studio downloads page <https://visualstudio.microsoft.com/downloads/>`_ by scrolling down to the 'All Downloads' section, clicking the disclosure triangle by 'Tools for Visual Studio 2022', then clicking the download button next to 'Build Tools for Visual Studio 2022'. When installing, choose the 'Desktop development with C++' Workload and select only 'MSVC v143' and 'Windows 10 SDK' or 'Windows 11 SDK options.
* Set the Path and Environment Variables - this can be done by following the `instructions from Microsoft <https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_file_locations>`_, or manually by adding a form of ``C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64`` (this may vary according to your exact machine and installation) to your system Path environment variable.

Alternatively, if you are using Windows 10/11 you can install the `Windows Subsystem for Linux <https://docs.microsoft.com/en-gb/windows/wsl/about>`_ and then follow the Linux install instructions for gprMax. Note however that currently, WSL does not aim to support GUI desktops or applications, e.g. Gnome, KDE, etc...


2. [Optional] Install MPI
--------------------------

MPI is required only for domain decomposition and task farming. Ordinary
serial, OpenMP, CUDA, OpenCL, and Metal simulations do not require an MPI
runtime or the ``mpi4py`` Python package. The ``mpi4py`` binding is installed
through the gprMax ``mpi`` extra in step 6.

If you plan to use MPI and are running gprMax on an HPC system, a suitable MPI
implementation will likely be installed already. Otherwise you will need to
install one yourself.

Linux/macOS
^^^^^^^^^^^
* It is recommended to use `OpenMPI <http://www.open-mpi.org>`_.

Microsoft Windows
^^^^^^^^^^^^^^^^^
* It is recommended to use `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_. Download and install both the .exe and .msi files.


3. Install Python, the required Python packages, and get the gprMax source
--------------------------------------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis. gprMax supports Python 3.11--3.13, and Python 3.12 is the recommended version for the first v4 release. The supplied ``conda_env.yml`` therefore creates a Python 3.12 environment.

* `Download and install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for your platform. The Python version used by the Miniconda installer does not need to match the Python 3.12 version selected by ``conda_env.yml``. We recommend choosing the installation options to: install Miniconda only for your user account; add Miniconda to your PATH environment variable; and register Miniconda Python as your default Python. See the `Quick Install page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for help installing Miniconda.
* Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:

.. code-block:: console

    $ conda update conda
    $ conda install git
    $ git clone https://github.com/gprMax/gprMax.git
    $ cd gprMax
    $ conda env create -f conda_env.yml

This will make sure conda is up-to-date, install Git, get the latest gprMax
source code from GitHub, and create an environment containing the core gprMax
packages. MPI is deliberately not installed in this base environment.

If you prefer to install Python and the required Python packages manually, i.e. without using Anaconda/Miniconda, look in the ``conda_env.yml`` file for a list of the requirements.

If you are using Arch Linux (https://www.archlinux.org/) you may need to also install ``wxPython`` by adding it to the conda environment file (``conda_env.yml``).


.. _h5py_mpi:

4. [Optional] Build h5py against Parallel HDF5
----------------------------------------------

If you plan to use the `MPI domain decomposition functionality
<https://docs.gprmax.com/en/latest/accelerators.html#mpi-domain-decomposition>`_
available in gprMax, h5py must be built with MPI support.

Install with conda
^^^^^^^^^^^^^^^^^^

h5py can be installed with MPI support in a conda environment with:

.. code:: console

    (gprMax)$ conda install -c conda-forge "h5py>=2.9=mpi*"

Install with pip
^^^^^^^^^^^^^^^^

Set your default compiler to the ``mpicc`` wrapper and build h5py with the ``HDF5_MPI`` environment variable:

.. code:: console

    (gprMax)$ export CC=mpicc
    (gprMax)$ export HDF5_MPI="ON"
    (gprMax)$ pip install --no-binary=h5py h5py  # Add --no-cache-dir if pip has cached a previous build of h5py

Further guidance on building h5py against a parallel build of HDF5 is available in the `h5py documentation <https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5>`_.


5. [Optional] Install mpi4py_fft
--------------------------------

If you plan to use the `MPI domain decomposition functionality
<https://docs.gprmax.com/en/latest/accelerators.html#mpi-domain-decomposition>`_
with fractal user objects, you need to install mpi4py_fft.

Python 3.12 is recommended for this optional configuration. ``mpi4py_fft``
contains Python-version-specific compiled extensions and also requires a
compatible MPI implementation and FFTW installation, so it is more sensitive
to the local software stack than the core gprMax package.

Install FFTW
^^^^^^^^^^^^

FFTW is a required dependency of mpi4py_fft, however, if you are running gprMax on a HPC system, FFTW may be available already - consult your site's documentation. Otherwise you will need to install it yourself.

Linux
#####

* It is possible binaries are available via your package manager. E.g. ``libfftw3-dev`` on Ubuntu.
* Otherwise you can find the latest source code on the `fftw downloads page <https://fftw.org/download.html>`_. There are instructions to build from source in the `fftw docs <https://fftw.org/fftw3_doc/Installation-on-Unix.html>`_.

macOS
#####

* FFTW can be installed using the `Homebrew package manager <http://brew.sh>`_:

.. code-block:: console

    $ brew install fftw

Microsoft Windows
#################

* While FFTW can be installed on Windows (guidance `here <https://fftw.org/install/windows.html>`_), it is not possible to build mpi4py_fft using the MSVC compiler.
* Therefore, we recommend using `Windows Subsystem for Linux <https://docs.microsoft.com/en-gb/windows/wsl/about>`_ and then following the Linux install instructions for gprMax.

Install with conda
^^^^^^^^^^^^^^^^^^

mpi4py_fft can be installed in a conda environment with:

.. code:: console

    (gprMax)$ conda install -c conda-forge mpi4py_fft

Install with pip
^^^^^^^^^^^^^^^^

mpi4py_fft can be installed using pip with:

.. code:: console

    (gprMax)$ pip install mpi4py_fft

.. tip::

    It may be necessary to tell mpi4py_fft where FFTW is installed. This can be done by setting the ``FFTW_INCLUDE_DIR`` and ``FFTW_LIBRARY_DIR`` environment variables to the appropriate paths.


6. Build and install gprMax
---------------------------

Once you have installed the aforementioned tools follow these steps to build and install gprMax:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), **navigate into the directory above the gprMax package**, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run the following commands:

.. code-block:: console

    (gprMax)$ pip install -e gprMax

Release and ordinary source builds use portable instruction sets. Developers
making a private build for the current host can opt into machine-specific
optimisation on Linux or macOS:

.. code-block:: console

    (gprMax)$ GPRMAX_BUILD_NATIVE=1 pip install -e gprMax

Do not redistribute a native build: it may contain instructions unavailable
on another processor. Published binary wheels never enable this option. Two
extension modules are compiled concurrently by default; constrained systems
can set ``GPRMAX_BUILD_JOBS=1``, while release builders may select a larger
positive value.

For MPI domain decomposition or task farming, install the MPI extra instead:

.. code-block:: console

    (gprMax)$ pip install -e "gprMax[mpi]"

For distributed fractal generation, use the combined extra:

.. code-block:: console

    (gprMax)$ pip install -e "gprMax[mpi-fractals]"

The ``mpi-fractals`` extra includes both ``mpi4py`` and ``mpi4py-fft``. If a
core source installation is already built, MPI support can be added without
recompiling gprMax by installing the system MPI runtime and then running
``python -m pip install mpi4py`` (plus ``mpi4py-fft`` for distributed
fractals). A compatible system MPI runtime is still required; the Python extra
does not install or configure that runtime.

Accelerator bindings are also optional and can be installed independently:

.. code-block:: console

    (gprMax)$ pip install -e "gprMax[cuda]"       # NVIDIA CUDA; Linux/Windows
    (gprMax)$ pip install -e "gprMax[opencl]"     # OpenCL
    (gprMax)$ pip install -e "gprMax[metal]"      # Apple Metal; macOS

Several extras can be requested together, for example
``gprMax[cuda,opencl]``. The ``gprMax[accelerators]`` convenience extra
requests every accelerator binding applicable to the current operating
system. It is not the default because package installers cannot detect whether
a compatible device, driver, CUDA toolkit, or OpenCL runtime is present, and
building an unavailable binding can prevent installation. The backend-specific
system software described in the `accelerator documentation
<https://docs.gprmax.com/en/latest/accelerators.html>`_ is still required.

The interactive Marimo dashboards have their own optional dependencies:

.. code-block:: console

    (gprMax)$ pip install -e "gprMax[marimo]"

See the `Marimo toolbox documentation
<https://docs.gprmax.com/en/latest/inc_Marimo.html>`_ for the available
A-scan, B-scan, progress, and introductory processing dashboards.

**You are now ready to proceed to running gprMax.**

Running gprMax
==============

gprMax is designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory.

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`.

Examples from a wheel installation
----------------------------------

Binary wheel installations also contain the examples that match the installed
gprMax version. Copy them from the read-only installation into a writable
workspace with:

.. code-block:: console

    (gprMax)$ python -m gprMax.examples list
    (gprMax)$ python -m gprMax.examples copy ~/gprMax-v4-examples
    (gprMax)$ cd ~/gprMax-v4-examples

The destination contains the normal ``examples/`` hierarchy, so the same
commands work for wheel and source installations. An existing example tree is
not overwritten unless ``--force`` is supplied.

Basic usage of gprMax is:

.. code-block:: console

    (gprMax)$ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: console

    (gprMax)$ python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in

To use Apple Metal GPU acceleration on macOS:

.. code-block:: bash

    (gprMax)$ python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in -metal

When the simulation is complete you can plot the A-scan using:

.. code-block:: console

    (gprMax)$ python -m toolboxes.Plotting.plot_Ascan examples/gpr/basic/cylinder_Ascan_2D.h5

Your results should be like those from the A-scan from the metal cylinder example in `introductory/basic 2D models section <http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results>`_

When you are finished using gprMax, the conda environment can be deactivated using :code:`conda deactivate`.

Optional command line arguments
-------------------------------

.. warning::

    ``-mpi`` has been depreciated in favour of ``--taskfarm``. Additionally, ``--mpi`` controls the new MPI domain decomposition functionality.

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
      - Number of required simulation runs. This option can be used to run a series of models, e.g. to create a B-scan with 60 traces: ``(gprMax)$ python -m gprMax examples/gpr/basic/cylinder_Bscan_2D.in -n 60``
    * - ``-i``
      - integer
      - Model number to start/restart the simulation from. It would typically be used to restart a series of models from a specific model number, with the n argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces.
    * - ``-t`` or ``--taskfarm``
      - flag
      - Flag to use Message Passing Interface (MPI) taskfarm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI taskfarm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``(gprMax)$ python -m gprMax examples/gpr/basic/cylinder_Bscan_2D.in -n 60 --taskfarm``. For further details see the
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
    * - ``-metal``
      - list/bool
      - Flag to use Apple Metal GPU or list of Metal device ID(s) for specific compute device(s) (macOS with M-series chips).
    * - ``-cpu_precision``
      - string
      - Precision for the CPU solver: ``single`` (default) or ``double``. This option is ignored when a GPU solver is used. Sub-gridding always uses double precision regardless of this setting.
    * - ``-gpu_precision``
      - string
      - Precision for the CUDA, OpenCL, or Metal solver: ``single`` (default) or ``double``. Apple Metal currently supports single precision only. This option is ignored when the CPU solver or sub-gridding is used.
    * - ``--geometry-only``
      - flag
      - Build a model and produce any geometry views but do not run the simulation, e.g. to check
        the geometry of a model is correct: ``(gprMax)$ python -m gprMax examples/gpr/materials/heterogeneous_soil.in --geometry-only``
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
      - Forces progress bars to be hidden - by default, progress bars are hidden when the log level is greater than info (20).
    * - ``--log-level``
      - integer
      - Level of logging to use, see the `Python logging module <https://docs.python.org/3/library/logging.html>`_.
    * - ``--log-file``
      - flag
      - Write logging information to file.
    * - ``--log-all-ranks``
      - flag
      - Write logging information from all MPI ranks. Default behaviour only provides log output
        from rank 0. When used with ``--log-file``, each rank will write to an individual file.
    * - ``-h`` or ``--help``
      - flag
      - Used to get help on command line options.

Updating gprMax
===============

* The safest and simplest way to upgrade gprMax is to uninstall, clone the latest version, and re-install the software. Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the directory above the gprMax package, and if it is not already active, activate the gprMax conda environment :code:`conda activate gprMax`. Run the following command:

.. code-block:: console

    (gprMax)$ pip uninstall gprMax
    (gprMax)$ git clone https://github.com/gprMax/gprMax.git
    (gprMax)$ pip install -e gprMax

This will uninstall gprMax, clone the most recent gprMax source code from GitHub, and then build and install the latest version of gprMax.


Updating conda and Python packages
----------------------------------

Periodically you should update conda and the required Python packages. With the gprMax environment deactivated and from the top-level gprMax directory, run the following commands:

.. code-block:: console

    $ conda update conda
    $ conda env update -f conda_env.yml


Thanks To Our Contributors ✨🔗
===============================
.. image:: https://contrib.rocks/image?repo=gprMax/gprMax
   :target: https://github.com/gprMax/gprMax/graphs/contributors
