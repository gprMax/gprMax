
***************
Getting Started
***************

What is gprMax?
===============

gprMax (http://www.gprmax.com) is free software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is released under the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).

gprMax is written in Python 3 (https://www.python.org) and includes performance-critical parts written in Cython/OpenMP (http://cython.org).

.. code-block:: none

    gprMax/
        gprMax/
        tests/
        tools/
        user_libs/
        user_models/
        README.rst
        setup.py

* ``gprMax`` is the main package. Within this package the main module is ``gprMax.py``
* ``tests`` is a sub-package which contains test modules and input files.
* ``tools`` is a sub-package which contains scripts to assist with viewing and post-processing output from models.
* ``user_libs`` is a sub-package where useful modules contributed by users are stored.
* ``user_models`` is a sub-package where useful input files contributed by users are stored.
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``setup.py`` is used to compile the Cython extension modules.

Installation
============

Get the code
------------

* Use **Git** (https://git-scm.com) and clone the master branch of the repository: :code:`git clone https://github.com/gprMax/gprMax.git`

* or **download a zip archive** of the code from https://github.com/gprMax/gprMax. Choose the ``Download ZIP`` button (right-hand side of the page).


Install Python and a C compiler
-------------------------------

To build and use the code you will need:

* **Python 3**.
* Python packages: **cython, h5py, matplotlib, numpy, psutil, pyfiglet**. Optionally **mpi4py** if you want to use the Message Passing Interface (MPI) task farm functionality (requires an installation of OpenMPI).
* **C compiler which supports OpenMP**

Use the following guidance dependent on your platform.

Mac OS X and Linux
^^^^^^^^^^^^^^^^^^

* Install Python 3 (https://www.python.org/downloads/)
* Install the aforementioned Python packages, which on Mac OS X can be done using the :code:`pip` package manager which comes with Python, e.g. :code:`pip install cython`. The same goes for Linux, or alternatively you can use the :code:`apt-get` package manager, e.g. :code:`sudo apt-get install python3-cython`. To check what packages are installed use :code:`pip list`.
* Install a C compiler which supports OpenMP. Linux should have gcc (https://gcc.gnu.org) already installed. With most recent versions of Mac OS X the LLVM (clang) is installed by default which does not support OpenMP. However, gcc is easily installed using the Homebrew package manager (http://brew.sh).

Microsoft Windows
^^^^^^^^^^^^^^^^^

Using the code on Windows is not as straightforward as for other platforms because of the combination of requirements. However, there are number of different ways of installing the required packages. We recommend you use the following procedure, as we have successfully tested it on Windows 7 (32/64-bit) and Windows 10 (64-bit). Please make sure you install the correct versions of binary packages depending on whether you have 32-bit or 64-bit Windows.

* Install Python 3 (https://www.python.org/downloads/)
* Download and install Microsoft Visual Studio 2015 Community (https://www.visualstudio.com/downloads/download-visual-studio-vs), which is free. Do a custom install and make sure under programming languages Visual C++ is selected, no other options are required.
* Create a new environment variable :code:`VS100COMNTOOLS` which matches the value of the existing :code:`VS140COMNTOOLS` environment variable. To set an environment variable from the Start Menu, right-click the Computer icon and select Properties. Click the Advanced System Settings link in the left column. In the System Properties window, click on the Advanced tab, then click the Environment Variables button near the bottom of that tab.
* Use the :code:`pip` package manager, which comes with Python, to install the cython, psutil, pyfiglet, pyparsing, python-dateutil, and pytz packages e.g. :code:`pip install cython`. To check what packages are installed use :code:`pip list`.
* Download binaries of packages numpy, h5py,  matplotlib (http://www.lfd.uci.edu/~gohlke/pythonlibs/) and install (in the aforementioned order) using ``pip``, e.g. :code:`pip install numpy-1.9.2+mkl-cp35-none-win_amd64.whl`

.. warning::

    If you use Anaconda, a popular Python distribution, please be aware that there is currently a bug with the HDF5 package (h5py) that is included with Anaconda (2.3.0). It effects 64-bit versions of Windows (https://github.com/h5py/h5py/issues/593). If you want to use Anaconda you should upgrade the h5py package by downloading and installing the correct binary from http://www.lfd.uci.edu/~gohlke/pythonlibs/, e.g. ``pip install --upgrade h5py‑2.5.0‑cp34‑none‑win_amd64.whl``


Compile Cython extensions
-------------------------

Once you have installed the aforementioned tools follow these steps to build the Cython extension modules for gprMax:

#. Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and navigate into the gprMax directory.
#. Compile the Cython extension modules using: :code:`python setup.py build_ext --inplace`. You should see a set of :code:`.c` source files and a set of :code:`.so` (Linux/Mac OS X) or :code:`.pyd` (Windows) compiled module files inside the gprMax directory.

.. note::

    If you want to remove/clean Cython generated files, e.g. before rebuilding the Cython extensions, you can use :code:`python setup.py cleanall`.

You are now ready to run gprMax.


Run the code
============

* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and navigate into the top-level gprMax directory. gprMax in designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory. Basic usage of gprMax is:

.. code-block:: none

    python -m gprMax path_to/name_of_input_file

For example to run one of the test models, navigate into the top-level gprMax directory and use:

.. code-block:: none

    python -m gprMax user_models/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: none

    python -m tools.plot_hdf5_Ascan user_models/cylinder_Ascan_2D.out

Optional command line arguments
-------------------------------

There are optional command line arguments for gprMax:

* ``--geometry-only`` will build a model and produce any geometry views but will not run the simulation. This option is useful for checking the geometry of the model is correct.
* ``-n`` is used along with a integer number to specify the number of times to run the input file. This option can be used to run a series of models, e.g. to create a B-scan that uses an antenna model.
* ``-mpi`` is a flag to turn on Message Passing Interface (MPI) task farm functionality. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using MPI. For further details see the :ref:`Parallel performance section <openmp_mpi>`.
* ``--commands-python`` will write an input file after any Python code blocks in the original input file have been processed.
* ``-h`` or ``--help`` can be used to get help on command line options.

For example, to check the geometry of a model:

.. code-block:: none

    python -m gprMax user_models/heterogeneous_soil.in --geometry-only

For example, to run a B-scan with 50 traces using MPI:

.. code-block:: none

    python -m gprMax user_models/GSSI_1500_cylinder_Bscan.in -n 54 -mpi




