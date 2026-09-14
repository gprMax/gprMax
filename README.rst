.. image:: https://readthedocs.org/projects/gprmax/badge/?version=latest
    :target: https://docs.gprmax.com/en/latest/?badge=latest
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

The subgridding lineage of gprMax includes the early `ADI-FDTD research of
Nectaria Diamanti <https://doi.org/10.1016/j.jappgeo.2008.07.004>`_. This work influenced the
subsequent `doctoral research of John Hartley
<https://era.ed.ac.uk/items/d253612b-7c1e-4adf-9a39-d730bbe76a95>`_, through
which version 4 was initiated and the current FDTD subgridding and
dispersive-interface averaging capabilities were developed. Nathan Mannall
subsequently carried out a major architectural refactoring and developed the
MPI domain-decomposition solver. Qifeng Shen developed the FDFD eigenmode
solver, wave-port and impedance-boundary capabilities, and has led the
expansion of antenna and RF modelling. Accelerator backends, testing
infrastructure, geometry importers, toolboxes, antenna models and documentation
also reflect substantial work by the wider gprMax community and Google Summer
of Code contributors. The `full authors and contributors record
<https://github.com/gprMax/gprMax/blob/master/AUTHORS.rst>`_ describes these
contributions in more detail.

After nearly three decades of development, the current version 4 codebase
extends substantially beyond the capabilities described in the 2016 paper.
Although GPR remains a core application, gprMax is now a `general-purpose
research platform for time-domain computational electromagnetics`. Applications
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

.. code-block:: text

    gprMax/
        .github/
        docs/
        examples/
        gprMax/
            toolboxes/
        images_shared/
        packaging/
        reframe_tests/
        testing/
        tests/
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
* ``docs/`` contains the source for the HTML and PDF versions of the User Guide. It uses
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
* ``gprMax/toolboxes/`` contains user-facing processing, conversion, visualisation,
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

.. _installation-pypi:

PyPI installation (recommended for running models)
--------------------------------------------------

For ordinary modelling and use of the toolboxes, install a released package
from the Python Package Index (PyPI). You do not need a source checkout or the
development environment. With a supported Python selected, the installation
command is:

.. code-block:: console

    $ python -m pip install gprMax

These PyPI commands apply after the v4 release is published. To install an
unpublished development version, use `Installing from source`_ instead.

Python 3.11--3.13 is supported, with Python 3.12 recommended. Release wheels
target 64-bit Linux x86-64, Windows x86-64, and Intel and Apple Silicon macOS.
A matching binary wheel contains the compiled CPU extensions: you do not
need a C compiler, OpenMP development headers, Git or Conda to install it.
pip installs the declared Python dependencies as well. If no compatible wheel
is available, pip may attempt a source build, which needs the build tools
described under `Installing from source`_. An optional accelerator binding
can also require its own compiler or runtime even when gprMax uses a wheel.

**A virtual environment is recommended but not required.** Choose one of the
two routes below. Use a new environment if you want to retain a v3 installation.

Which Python does pip use?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Install a supported Python with pip first, for example using the
`Python downloads <https://www.python.org/downloads/>`_ or your platform's
Python provider. Installing gprMax with pip does not install Python itself.

``python -m pip`` installs into the Python selected by ``python``; it does not
install gprMax into every Python on the computer. A terminal with an activated
Conda environment uses that environment's Python, even though the installer
is pip. ``python`` may be called ``python3`` or ``python3.12`` on your system;
on Windows, ``py -3.12`` can select an installed Python 3.12. Use the same
interpreter for installation and execution.

If you are unsure which interpreter is selected, check it before installing:

.. code-block:: console

    $ python -c "import sys; print(sys.executable); print(sys.version)"
    $ python -m pip --version

These are setup/troubleshooting checks, not commands required before each
simulation. See the `pip user guide <https://pip.pypa.io/en/stable/user_guide/>`_
for how interpreter selection works.

Option A: use a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``venv`` keeps gprMax and its dependencies separate from other Python
projects. With a separately installed Python 3.12, create a new working folder
and environment. Do not create it inside your old v3 checkout.

Linux/macOS (bash or zsh):

.. code-block:: console

    $ mkdir gprmax-work
    $ cd gprmax-work
    $ python3.12 -m venv .venv
    $ source .venv/bin/activate
    $ python -m pip install gprMax
    $ python -m gprMax --help

Windows (PowerShell):

.. code-block:: powershell

    mkdir gprmax-work
    cd gprmax-work
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install gprMax
    python -m gprMax --help

In a later terminal, return to ``gprmax-work`` and activate the existing
environment again; do not recreate or reinstall it for each simulation.
Alternatively, activation is optional if you call its Python explicitly:

.. code-block:: console

    $ .venv/bin/python -m gprMax model.in

On Windows, the equivalent is ``.\.venv\Scripts\python.exe -m gprMax model.in``.
This also avoids changing PowerShell execution policy if activation scripts
are blocked. Use ``deactivate`` when finished with an activated ``venv``.
See the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_
for other shells. If Conda is already active and you want a non-Conda setup,
deactivate it and select your separately installed Python explicitly.

Option B: install without a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a user-managed Python installation that permits pip package installation,
you can install and run directly, with no activation step:

.. code-block:: console

    $ python -m pip install gprMax
    $ python -m gprMax model.in

Here ``model.in`` is your input file. This route shares dependencies with
other packages in that Python installation. Updates may affect those projects,
and v3 and v4 are not independently selectable installations in the same
environment. If that Python already contains v3, use a separate environment
to preserve it rather than installing over it.

If pip reports ``externally-managed-environment``, the OS or another package
manager owns that Python. Use Option A or a separate user-managed Python.
Do not use ``sudo pip`` or ``--break-system-packages`` as a routine workaround:
overriding those safeguards can interfere with software managed by the OS.
``--user`` is not an isolation mechanism and may also be blocked. See the
`externally managed environments guidance
<https://packaging.python.org/en/latest/specifications/externally-managed-environments/>`_.

If installation fails because the destination is not writable, create a
``venv`` in a writable directory rather than elevating permissions.

Working directories and installed files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run models and plotting commands from your own working directory. Neither a
wheel installation nor its toolboxes require you to enter ``site-packages``.
Keep inputs, scripts and outputs outside the installed package so that an
upgrade does not overwrite your work. See `Running gprMax`_ for a complete
example and `Using the installed toolboxes`_ for toolbox commands and assets.

There is no single installation path: it depends on the selected Python.
For the Linux/macOS environment above it is normally
``gprmax-work/.venv/lib/python3.12/site-packages/gprMax/``; on Windows it is
``gprmax-work\.venv\Lib\site-packages\gprMax\``. Locate the actual installation
with:

.. code-block:: console

    $ python -c "import gprMax; print(gprMax.__file__)"

Do not run v4 from a directory containing an old v3 ``gprMax`` package, or add
the old checkout to ``PYTHONPATH``. A local package can shadow the installed
one, even after changing environments. Select the same environment in your
IDE or notebook kernel as in the terminal.

Optional features
^^^^^^^^^^^^^^^^^

Install only the optional features you need, into the same selected Python:

.. code-block:: console

    $ python -m pip install "gprMax[mpi]"
    $ python -m pip install "gprMax[cuda]"
    $ python -m pip install "gprMax[opencl]"
    $ python -m pip install "gprMax[metal]"

These extras install Python bindings only. MPI, CUDA, OpenCL, and Metal still
require compatible system runtimes and hardware as described below and in
the `accelerator documentation
<https://docs.gprmax.com/en/latest/accelerators.html>`_.

.. _installation-conda:

Conda environments and coexistence with v3
------------------------------------------

Conda is useful for source development, testing and managing native libraries,
but it is not a gprMax requirement. It can also provide an isolated environment
for an ordinary PyPI installation. Choose Conda **instead of** the ``venv``
route; there is no need to nest them for normal use.

To run a released v4 with Conda, create a new environment:

.. code-block:: console

    $ conda create --name gprMax-v4 python=3.12 pip
    $ conda activate gprMax-v4
    $ python -m pip install gprMax
    $ python -m gprMax --help

This uses Conda for Python and pip for gprMax; it does not need the repository's
larger development environment. In a new terminal, activate ``gprMax-v4``
before running models. Alternatively, select the environment for one command:

.. code-block:: console

    $ conda run --no-capture-output -n gprMax-v4 python -m gprMax model.in

Use ``conda deactivate`` when finished with an activated Conda environment.
See `Conda environment management
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
and `conda run <https://docs.conda.io/projects/conda/en/stable/commands/run.html>`_.

``gprMax-v4`` is the default environment name for both released-package and
source installations: it identifies the major version, not a Git branch.
Choose one installation route; do not run both environment-creation commands
for the same name. If you need a wheel installation and an editable checkout
side by side, give the second environment a distinct name with ``--name``
(for example, ``gprMax-v4-source``) and activate that name instead.

.. _installation-coexistence:

Keeping v3 and v4 side by side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the old environment and, for an editable installation, its source
checkout. Do not rename, delete or update that checkout to v4: the v3
environment may import directly from it. Use a separate checkout for v4
development and separate working/output directories for comparisons.

If the original v3 environment is named ``gprMax``, select versions with:

.. code-block:: console

    $ conda activate gprMax-v4
    $ python -m gprMax model_v4.in
    $ conda activate gprMax
    $ python -m gprMax model_v3.in

Replace ``gprMax`` with the actual old environment name. There is no
machine-wide "default gprMax" that permanently wins: the selected interpreter
and its import paths determine the version. Both versions use the package
name ``gprMax``. Do not install v4 or apply the v4 ``conda_env.yml`` to the old
environment if you want v3 to remain available; avoid using Conda's ``base``
environment for either project.

For a one-time coexistence check, run this outside both source checkouts:

.. code-block:: console

    $ python -c "import sys, gprMax; print(sys.executable); print(gprMax.__version__); print(gprMax.__file__)"

Installing v4 does not itself convert v3 inputs or results. Read the
`v3-to-v4 migration guide
<https://docs.gprmax.com/en/latest/migration_v3_v4.html>`_ before comparing them.

.. _installation-source:

Installing from source
----------------------

Build from source when developing gprMax, modifying its Cython extensions, or
using a platform for which no wheel is published. The supplied
``conda_env.yml`` is development-focused: it includes notebooks, tests and
optional geometry/visualisation packages such as pythonocc-core and PyVista.
It is larger than the dependencies of a core PyPI installation and is not
needed merely to run a released wheel. On HPC systems, use your site's
compiler/MPI modules and consult the `HPC guide
<https://docs.gprmax.com/en/latest/hpc.html>`_.

Use a separate source checkout and environment, as below. The source-build
steps are:

1. Install a C compiler which supports OpenMP
2. [Optional] Install MPI
3. Install Python, required Python packages, and get the gprMax source code from GitHub
4. [Optional] Build h5py against Parallel HDF5
5. [Optional] Install mpi4py_fft
6. Build and install gprMax

1. Install a C compiler which supports OpenMP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux
"""""

* Install `GCC <https://gcc.gnu.org>`_ and the corresponding development
  tools using your distribution's package manager, or load the compiler
  module on an HPC system. They may already be available; check ``gcc --version``.


macOS
"""""

* Install the Xcode command-line tools, which provide Apple Clang, and install
  the OpenMP runtime using `Homebrew <https://brew.sh>`_:

.. code-block:: console

    $ xcode-select --install
    $ brew install libomp

  gprMax detects ``libomp`` from Homebrew, the active Python environment, or
  ``GPRMAX_LIBOMP_PREFIX``. The previous Homebrew GCC requirement is no longer
  necessary.

Microsoft Windows
"""""""""""""""""

* Download and install Microsoft `Build Tools for Visual Studio 2022 <https://aka.ms/vs/17/release/vs_BuildTools.exe>`_ (direct link). You can also find it on the `Microsoft Visual Studio downloads page <https://visualstudio.microsoft.com/downloads/>`_ by scrolling down to the 'All Downloads' section, clicking the disclosure triangle by 'Tools for Visual Studio 2022', then clicking the download button next to 'Build Tools for Visual Studio 2022'. When installing, choose the 'Desktop development with C++' Workload and select only 'MSVC v143' and 'Windows 10 SDK' or 'Windows 11 SDK options.
* Use an **x64 Native Tools Command Prompt for Visual Studio**, then activate
  the chosen Python environment there. The compiler needs its headers and
  libraries as well as its executable; adding a guessed compiler directory
  to ``PATH`` is not sufficient. See the `Microsoft compiler-environment
  instructions <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line>`_.

Alternatively, use the `Windows Subsystem for Linux
<https://learn.microsoft.com/en-us/windows/wsl/>`_ and follow the Linux
instructions. That is a separate Linux Python installation, not the Windows one.


2. [Optional] Install MPI
^^^^^^^^^^^^^^^^^^^^^^^^^

MPI is required only for domain decomposition and task farming. Ordinary
serial, OpenMP, CUDA, OpenCL, and Metal simulations do not require an MPI
runtime or the ``mpi4py`` Python package. The ``mpi4py`` binding is installed
through the gprMax ``mpi`` extra in step 6.

If you plan to use MPI and are running gprMax on an HPC system, a suitable MPI
implementation will likely be installed already. Otherwise you will need to
install one yourself.

Linux/macOS
"""""""""""
* It is recommended to use `OpenMPI <http://www.open-mpi.org>`_.

Microsoft Windows
"""""""""""""""""
* It is recommended to use `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_. Download and install both the .exe and .msi files.


3. Install Python, the required Python packages, and get the gprMax source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install `Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
if it is not already available, and install `Git <https://git-scm.com/downloads>`_.
There is no need to replace an existing Conda installation or make its Python
the system-wide default. The Python in Conda's base environment can differ
from Python 3.12 in the development environment.

The following commands select the ``devel`` branch for v4 development, not a
pinned release. To build a published release instead, check out that release's
tag in the new checkout before creating its environment. Use a new directory
and environment name; do not reuse your v3 checkout or environment:

.. code-block:: console

    $ git clone --branch devel https://github.com/gprMax/gprMax.git gprMax-v4-src
    $ cd gprMax-v4-src
    $ conda env create -f conda_env.yml
    $ conda activate gprMax-v4

The YAML file defaults to ``gprMax-v4``. This step installs development
dependencies, not gprMax itself; step 6 builds and installs the checkout.
MPI and accelerator bindings remain optional. If ``gprMax-v4`` already exists,
keep it and choose a different name with ``--name`` rather than overwriting it.

For a non-Conda source environment, create and activate a ``venv`` using the
earlier instructions, then run ``python -m pip install -r requirements.txt``
from the checkout for the common development/test dependencies. Optional CAD,
MPI and accelerator libraries still need their own installation instructions.
For a minimal source installation, step 6 installs the declared package
dependencies and isolated build requirements without that development list.


.. _h5py_mpi:

4. [Optional] Build h5py against Parallel HDF5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to use parallel HDF5 output with the `MPI domain decomposition functionality
<https://docs.gprmax.com/en/latest/accelerators.html#mpi-domain-decomposition>`_
available in gprMax, h5py must be built with MPI support.

Install with conda
""""""""""""""""""

h5py can be installed with MPI support in a conda environment with:

.. code:: console

    $ conda install -c conda-forge "h5py>=2.9=mpi*"

Install with pip
""""""""""""""""

For a pip-managed environment on Linux/macOS, install a compatible parallel
HDF5 library and h5py's dependencies first, then build with ``mpicc`` and
``HDF5_MPI``. If serial h5py is already installed, an ordinary pip install
can report it as already satisfied instead of rebuilding it:

.. code:: console

    $ CC=mpicc HDF5_MPI=ON python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=h5py h5py

``--no-deps`` leaves the already prepared runtime dependencies unchanged.
For a Conda-managed h5py installation, prefer the Conda route above rather
than replacing its files with pip. Verify MPI support in the selected Python:

.. code-block:: console

    $ python -c "import h5py; print(h5py.get_config().mpi)"

Further guidance on building h5py against a parallel build of HDF5 is available in the `h5py documentation <https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5>`_.


5. [Optional] Install mpi4py_fft
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to use the `MPI domain decomposition functionality
<https://docs.gprmax.com/en/latest/accelerators.html#mpi-domain-decomposition>`_
with fractal user objects, you need to install mpi4py_fft.

Python 3.12 is recommended for this optional configuration. ``mpi4py_fft``
contains Python-version-specific compiled extensions and also requires a
compatible MPI implementation and FFTW installation, so it is more sensitive
to the local software stack than the core gprMax package.

Install FFTW
""""""""""""

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
* Therefore, we recommend using `Windows Subsystem for Linux`_ and then
  following the Linux install instructions for gprMax.

Install with conda
""""""""""""""""""

mpi4py_fft can be installed in a conda environment with:

.. code:: console

    $ conda install -c conda-forge mpi4py_fft

Install with pip
""""""""""""""""

mpi4py_fft can be installed using pip with:

.. code:: console

    $ python -m pip install mpi4py_fft

.. tip::

    It may be necessary to tell mpi4py_fft where FFTW is installed. This can be done by setting the ``FFTW_INCLUDE_DIR`` and ``FFTW_LIBRARY_DIR`` environment variables to the appropriate paths.


6. Build and install gprMax
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stay in the top-level ``gprMax-v4-src`` checkout (the directory containing
``setup.py`` and ``pyproject.toml``), with ``gprMax-v4`` or your chosen
source-build environment active:

.. code-block:: console

    $ python -m pip install -e .

The ``.`` means this checkout, and ``-e`` makes the installation editable.
Python changes are read from these files; changes to compiled Cython code
require rebuilding with the same command. Do not delete or move the checkout
while using its editable installation. For a non-editable source install,
use ``python -m pip install .`` instead.

Release and ordinary source builds use portable instruction sets. Developers
making a private build for the current host can opt into machine-specific
optimisation on Linux or macOS:

.. code-block:: console

    $ GPRMAX_BUILD_NATIVE=1 python -m pip install -e .

Do not redistribute a native build: it may contain instructions unavailable
on another processor. Published binary wheels never enable this option. Two
extension modules are compiled concurrently by default; constrained systems
can set ``GPRMAX_BUILD_JOBS=1``, while release builders may select a larger
positive value.

For MPI domain decomposition or task farming, install the MPI extra instead:

.. code-block:: console

    $ python -m pip install -e ".[mpi]"

For distributed fractal generation, use the combined extra:

.. code-block:: console

    $ python -m pip install -e ".[mpi-fractals]"

The ``mpi-fractals`` extra includes both ``mpi4py`` and ``mpi4py-fft``. If a
core source installation is already built, MPI support can be added without
recompiling gprMax by installing the system MPI runtime and then running
``python -m pip install mpi4py`` (plus ``mpi4py-fft`` for distributed
fractals). A compatible system MPI runtime is still required; the Python extra
does not install or configure that runtime.

Accelerator bindings are also optional and can be installed independently:

.. code-block:: console

    $ python -m pip install -e ".[cuda]"       # NVIDIA CUDA; Linux/Windows
    $ python -m pip install -e ".[opencl]"     # OpenCL
    $ python -m pip install -e ".[metal]"      # Apple Metal; macOS

These editable targets refer to the current checkout. In contrast,
``python -m pip install "gprMax[cuda]"`` selects the published distribution;
do not substitute that command when you mean to use your edited source.

Several extras can be requested together, for example
``.[cuda,opencl]``. The ``.[accelerators]`` convenience extra
requests every accelerator binding applicable to the current operating
system. It is not the default because package installers cannot detect whether
a compatible device, driver, CUDA toolkit, or OpenCL runtime is present, and
building an unavailable binding can prevent installation. The backend-specific
system software described in the `accelerator documentation
<https://docs.gprmax.com/en/latest/accelerators.html>`_ is still required.

The interactive Marimo dashboards have their own optional dependencies:

.. code-block:: console

    $ python -m pip install -e ".[marimo]"

See the `Marimo toolbox documentation
<https://docs.gprmax.com/en/latest/inc_Marimo.html>`_ for the available
A-scan, B-scan, progress, and introductory processing dashboards.

**You are now ready to proceed to running gprMax.**

Running gprMax
==============

gprMax is designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory.

Open a terminal and select the Python where gprMax is installed. If using an
environment, activate it or explicitly select its interpreter; a direct
installation into a user-managed Python needs no activation.
Run models from your own working directory; a wheel installation does not
require a repository checkout. Source users can also work from the repository
root for the version they intend to run. The ``$`` below represents a shell
prompt, not a required environment name; do not type it.

Using the installed toolboxes
-----------------------------

Toolboxes are included under the ``gprMax.toolboxes`` namespace, so they do
not compete with an unrelated Python package called ``toolboxes``. Run a
toolbox from your working directory, for example:

.. code-block:: console

    python -m gprMax.toolboxes.Plotting.plot_Ascan model.h5
    python -m gprMax.toolboxes.Plotting.plot_port model.h5 --list-ports

Python model scripts import the same namespace:

.. code-block:: python

    from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

To locate toolbox-specific examples, MATLAB utilities or other bundled files:

.. code-block:: console

    python -c "from importlib.resources import files; print(files('gprMax.toolboxes'))"

Copy files you want to edit into your own workspace; do not modify the
installed package. The example-copy command below copies the main model
examples, not all toolbox folders. Some specialist toolboxes need additional
optional dependencies, and large reference datasets may remain in the source
repository; consult the selected toolbox's documentation.

Existing scripts must change ``toolboxes.<name>`` to
``gprMax.toolboxes.<name>`` in imports and ``python -m`` commands. No top-level
``toolboxes`` compatibility package is installed, because that would retain
the naming conflict. An isolated environment is recommended when upgrading
an older development installation that installed the former namespace.

Examples from a wheel installation
----------------------------------

Binary wheel installations also contain the examples that match the installed
gprMax version. Copy them from the installed package into a writable
workspace with:

.. code-block:: console

    $ python -m gprMax.examples list
    $ python -m gprMax.examples copy ~/gprMax-v4-examples
    $ cd ~/gprMax-v4-examples

The destination contains the normal ``examples/`` hierarchy, so the same
commands work for wheel and source installations. An existing example tree is
not overwritten unless ``--force`` is supplied.

Basic usage of gprMax is:

.. code-block:: console

    $ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: console

    $ python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in

To use Apple Metal GPU acceleration on macOS:

.. code-block:: bash

    $ python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in -metal

When the simulation is complete you can plot the A-scan using:

.. code-block:: console

    $ python -m gprMax.toolboxes.Plotting.plot_Ascan examples/gpr/basic/cylinder_Ascan_2D.h5

Your results should be like those from the A-scan from the metal cylinder example in `introductory/basic 2D models section <http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results>`_

When finished, use ``deactivate`` for an activated ``venv`` or
``conda deactivate`` for Conda. No deactivation is needed when you selected
the interpreter directly without activating an environment.

Optional command line arguments
-------------------------------

.. warning::

    ``-mpi`` has been deprecated in favour of ``--taskfarm``. Additionally, ``--mpi`` controls the new MPI domain decomposition functionality.

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
      - Number of required simulation runs. This option can be used to run a series of models, e.g. to create a B-scan with 60 traces: ``$ python -m gprMax examples/gpr/basic/cylinder_Bscan_2D.in -n 60``
    * - ``-i``
      - integer
      - Model number to start/restart the simulation from. It would typically be used to restart a series of models from a specific model number, with the n argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces.
    * - ``-t`` or ``--taskfarm``
      - flag
      - Flag to use Message Passing Interface (MPI) taskfarm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI taskfarm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``$ python -m gprMax examples/gpr/basic/cylinder_Bscan_2D.in -n 60 --taskfarm``. For further details see the
        `MPI task-farm section of the User Guide <https://docs.gprmax.com/en/latest/accelerators.html#task-farm>`_
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
        the geometry of a model is correct: ``$ python -m gprMax examples/gpr/materials/heterogeneous_soil.in --geometry-only``
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

Updating a released v4 installation
-----------------------------------

Select the environment you intend to update, then use pip. No Git checkout or
manual uninstallation is needed:

.. code-block:: console

    $ python -m pip install --upgrade "gprMax>=4,<5"

The version range keeps an existing v4 project on the v4 release series.
An unversioned ``pip install gprMax`` may report an existing installation as
already satisfied; it is not an explicit upgrade request. For reproducible
work, record or pin the exact version and retain representative reference
outputs before updating. This command updates the selected environment, not
every installation on the machine.

For v3-to-v4 migration, create a separate environment first as described in
`Keeping v3 and v4 side by side`_. Do not use this upgrade command inside the
old v3 environment if you want to retain that installation.

Updating a source-development installation
------------------------------------------

Preserve or commit your local edits before switching branches/tags or pulling
changes. In the separate v4 checkout, select the desired revision, activate
its development environment, then rebuild with ``python -m pip install -e .``.
Do not use a PyPI upgrade command to update an editable checkout.

If the revision changes dependencies, recreate a development environment
from its YAML file under a new name, for example:

.. code-block:: console

    $ conda env create --name gprMax-v4-next --file conda_env.yml
    $ conda activate gprMax-v4-next
    $ python -m pip install -e .

Reinstall any selected optional bindings. Install Conda-managed dependencies
before the final pip installation; repeated Conda/pip changes in place can
leave inconsistent dependency sets. Do not apply this YAML file to a v3
environment or to Conda's ``base``. The environment name is only a selector:
an editable install still follows the source files in its checkout.


Thanks to our contributors
==========================

The complete and current contributor history is available in the `GitHub
contributor graph <https://github.com/gprMax/gprMax/graphs/contributors>`_.
