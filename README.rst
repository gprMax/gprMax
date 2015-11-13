
***************
Getting Started
***************

What is gprMax?
===============

gprMax (http://www.gprmax.com) is free software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is released under the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).

gprMax is written in Python 3 (https://www.python.org) and includes performance-critical parts written in Cython/OpenMP (http://cython.org).

Using gprMax? Cite us.
----------------------

If you use gprMax and publish your work we would be grateful if you could cite gprMax using the following references:

* Warren, C., Giannopoulos, A., & Giannakis I. (2015). An advanced GPR modelling framework – the next generation of gprMax, In `Proc. 8th Int. Workshop Advanced Ground Penetrating Radar` (http://dx.doi.org/10.1109/IWAGPR.2015.7292621)
* Giannopoulos, A. (2005). Modelling ground penetrating radar by GprMax, `Construction and Building Materials`, 19(10), 755-762 (http://dx.doi.org/10.1016/j.conbuildmat.2005.06.007)

Software structure
==================

.. code-block:: none

    gprMax/
        docs/
        gprMax/
        LICENSE
        README.rst
        setup.py
        tests/
        tools/
        user_libs/
        user_models/


* ``docs`` contains source files for the User Guide. The User Guide is written using reStructuredText (http://docutils.sourceforge.net/rst.html) markup, and is built using Sphinx (http://sphinx-doc.org) and Read the Docs (https://readthedocs.org).
* ``gprMax`` is the main package. Within this package the main module is ``gprMax.py``
* ``LICENSE`` contains information on the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``setup.py`` is used to compile the Cython extension modules.
* ``tests`` is a sub-package which contains test modules and input files.
* ``tools`` is a sub-package which contains scripts to assist with viewing and post-processing output from models.
* ``user_libs`` is a sub-package where useful modules contributed by users are stored.
* ``user_models`` is a sub-package where useful input files contributed by users are stored.

Installation (End users)
========================

You should use the following guidance if you intend to use gprMax `without` developing the code. There are 2 main steps to the installation: Getting the code; and installing Python, packages, and C libraries.

Get the code
------------

* Download the code from https://github.com/gprMax/gprMax

    * Click on `Releases` from the top header and choose the release you want (latest is at the top).
    * Download zip files of the `source code` and `binary extensions` for your platform (``windows`` for 64-bit versions of Microsoft Windows or ``linux_macosx`` for 64-bit versions of Linux or Mac OS X).
    * Expand both zip files and copy the contents of the ``windows`` or ``linux_macosx`` directory into the ``gprMax-v.X.Y.Z/gprMax`` directory.

Install Python, packages, and C libraries
-----------------------------------------

Mac OS X and Linux
^^^^^^^^^^^^^^^^^^

* Install Python 3 (https://www.python.org/downloads/)
* Install Python packages: **cython, h5py, matplotlib, numpy, psutil, pyfiglet**. Optionally **mpi4py** if you want to use the Message Passing Interface (MPI) task farm functionality (requires an installation of OpenMPI).

    * Mac OS X: packages can be installed using the :code:`pip` package manager which comes with Python, e.g. :code:`pip install cython`. To check what packages are installed use :code:`pip list`.
    * Linux: packages can be installed using the :code:`pip` package manager which comes with Python, e.g. :code:`pip install cython`, or alternatively using the :code:`apt-get` package manager, e.g. :code:`sudo apt-get install python3-cython`. To check what packages are installed use :code:`pip list`.

* Install/Check C libraries:

    * Mac OS X: gcc is easily installed using the Homebrew package manager (http://brew.sh). Installations of Xcode on Mac OS X come with the LLVM (clang) compiler, but it does not currently support OpenMP, so you must install gcc.
    * Linux: gcc (https://gcc.gnu.org) should be already installed, so no action is required.


Microsoft Windows
^^^^^^^^^^^^^^^^^

* Install Python 3 (https://www.python.org/downloads/)
* Install Python packages:

    * Use the :code:`pip` package manager, which comes with Python, to install the **cython, pyfiglet, pyparsing, python-dateutil, and pytz** packages, e.g. :code:`pip install cython`. To check what packages are installed use :code:`pip list`.
    * Download binaries of packages **psutil, numpy, h5py,  matplotlib** from http://www.lfd.uci.edu/~gohlke/pythonlibs/, and then install (in the aforementioned order) using ``pip``, e.g. :code:`pip install numpy-1.9.2+mkl-cp35-none-win_amd64.whl`

* Install the Microsoft Visual Studio 2015 C++ Redistributable (``vc_redist.x64.exe``) from https://www.microsoft.com/en-us/download/details.aspx?id=48145.

**You are now ready to proceed to the section on running gprMax.**


Installation (Developers)
=========================

You should use the following guidance if you intend to develop the gprMax code. There are 3 main steps to the installation: Getting the code; installing Python, packages, and a C compiler; and compiling the Cython extensions.

Get the code
------------

* Use **Git** (https://git-scm.com) and clone the master branch of the repository: :code:`git clone https://github.com/gprMax/gprMax.git`
* or **download a zip archive** of the code from https://github.com/gprMax/gprMax. Choose the ``Download ZIP`` button (right-hand side of the page).


Install Python, packages, and a C compiler
------------------------------------------

Mac OS X and Linux
^^^^^^^^^^^^^^^^^^

* Install Python 3 (https://www.python.org/downloads/)
* Install Python packages: **cython, h5py, matplotlib, numpy, psutil, pyfiglet**. Optionally **mpi4py** if you want to use the Message Passing Interface (MPI) task farm functionality (requires an installation of OpenMPI).

    * Mac OS X: packages can be installed using the :code:`pip` package manager which comes with Python, e.g. :code:`pip install cython`. To check what packages are installed use :code:`pip list`.
    * Linux: packages can be installed using the :code:`pip` package manager which comes with Python, e.g. :code:`pip install cython`, or alternatively using the :code:`apt-get` package manager, e.g. :code:`sudo apt-get install python3-cython`. To check what packages are installed use :code:`pip list`.

* Install a C compiler which supports OpenMP:

    * Mac OS X: gcc is easily installed using the Homebrew package manager (http://brew.sh). Installations of Xcode on Mac OS X come with the LLVM (clang) compiler, but it does not currently support OpenMP, so you must install gcc.
    * Linux: gcc (https://gcc.gnu.org) should be already installed, so no action is required.


Microsoft Windows
^^^^^^^^^^^^^^^^^

* Install Python 3 (https://www.python.org/downloads/)
* Install a C compiler which supports OpenMP:

    * Download and install Microsoft Visual Studio 2015 Community (https://www.visualstudio.com/downloads/download-visual-studio-vs), which is free. Do a custom install and make sure under programming languages Visual C++ is selected, no other options are required.
    * Create a new environment variable :code:`VS100COMNTOOLS` which matches the value of the existing :code:`VS140COMNTOOLS` environment variable. To set an environment variable from the Start Menu, right-click the Computer icon and select Properties. Click the Advanced System Settings link in the left column. In the System Properties window, click on the Advanced tab, then click the Environment Variables button near the bottom of that tab.

* Install Python packages:

    * Use the :code:`pip` package manager, which comes with Python, to install the **cython, pyfiglet, pyparsing, python-dateutil, and pytz** packages, e.g. :code:`pip install cython`. To check what packages are installed use :code:`pip list`.
    * Download binaries of packages **psutil, numpy, h5py,  matplotlib** from http://www.lfd.uci.edu/~gohlke/pythonlibs/, and then install (in the aforementioned order) using ``pip``, e.g. :code:`pip install numpy-1.9.2+mkl-cp35-none-win_amd64.whl`

.. warning::

    If you use Anaconda, a popular Python distribution, please be aware that there is currently a bug with the HDF5 package (h5py) that is included with Anaconda (2.3.0). It effects 64-bit versions of Microsoft Windows (https://github.com/h5py/h5py/issues/593). If you want to use Anaconda you should upgrade the h5py package by downloading and installing the correct binary from http://www.lfd.uci.edu/~gohlke/pythonlibs/, e.g. ``pip install --upgrade h5py‑2.5.0‑cp34‑none‑win_amd64.whl``


Compile Cython extensions
-------------------------

Once you have installed the aforementioned tools follow these steps to build the Cython extension modules for gprMax:

#. Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and navigate into the gprMax directory.
#. Compile the Cython extension modules using: :code:`python setup.py build_ext --inplace`. You should see a set of :code:`.c` source files and a set of :code:`.so` (Linux/Mac OS X) or :code:`.pyd` (Windows) compiled module files inside the gprMax directory.

.. note::

    If you want to remove/clean Cython generated files, e.g. before rebuilding the Cython extensions, you can use :code:`python setup.py cleanall`.

**You are now ready to proceed to the section on running gprMax.**


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

    python -m tools.plot_Ascan_hdf5 user_models/cylinder_Ascan_2D.out

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

For example, to run a B-scan with 54 traces:

.. code-block:: none

    python -m gprMax user_models/GSSI_1500_cylinder_Bscan.in -n 54




