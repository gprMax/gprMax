.. image:: https://readthedocs.org/projects/gprmax/badge/?version=latest
    :target: http://docs.gprmax.com/en/latest/?badge=latest
    :alt: Documentation Status

***************
Getting Started
***************

What is gprMax?
===============

gprMax (http://www.gprmax.com) is open source software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is currently released under the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).

gprMax is written in Python 3 (https://www.python.org) and includes performance-critical parts written in Cython/OpenMP (http://cython.org).

Using gprMax? Cite us
---------------------

If you use gprMax and publish your work we would be grateful if you could cite gprMax using the following references:

* Warren, C., Giannopoulos, A., & Giannakis I. (2015). An advanced GPR modelling framework – the next generation of gprMax, In `Proc. 8th Int. Workshop Advanced Ground Penetrating Radar` (http://dx.doi.org/10.1109/IWAGPR.2015.7292621)
* Giannopoulos, A. (2005). Modelling ground penetrating radar by GprMax, `Construction and Building Materials`, 19(10), 755-762 (http://dx.doi.org/10.1016/j.conbuildmat.2005.06.007)

Package overview
================

.. code-block:: none

    gprMax/
        conda_env.yml
        CONTRIBUTORS
        docs/
        gprMax/
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
* ``docs`` contains source files for the User Guide. The User Guide is written using reStructuredText (http://docutils.sourceforge.net/rst.html) markup, and is built using Sphinx (http://sphinx-doc.org) and Read the Docs (https://readthedocs.org).
* ``gprMax`` is the main package. Within this package the main module is ``gprMax.py``
* ``LICENSE`` contains information on the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``setup.cfg`` is used to set preference for code formatting/styling using flake8.
* ``setup.py`` is used to compile the Cython extension modules.
* ``tests`` is a sub-package which contains test modules and input files.
* ``tools`` is a sub-package which contains scripts to assist with viewing and post-processing output from models.
* ``user_libs`` is a sub-package where useful modules contributed by users are stored.
* ``user_models`` is a sub-package where useful input files contributed by users are stored.

Installation
============

The following steps provide guidance on how to install gprMax:

1. Install Python, required Python packages, and get the gprMax source code from GitHub
2. Install a C compiler which supports OpenMP
3. Build and install gprMax

1. Install Python, required Python packages, and get gprMax source
------------------------------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis.

* Install Miniconda (Python 3.5 version) from http://conda.pydata.org/miniconda.html (help with Miniconda installation from http://conda.pydata.org/docs/install/quick.html)
* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and run the following commands:

.. code-block:: none

    $ conda update conda
    $ conda install git
    $ git clone https://github.com/gprMax/gprMax.git
    $ cd gprMax
    $ conda env create -f conda_env.yml

This will make sure conda is up-to-date, install Git, get the latest gprMax source code from GitHub, and create an environment for gprMax with all the necessary Python packages.

If you prefer to install Python and the required Python packages manually, i.e. without using Anaconda/Miniconda, look in the ``conda_env.yml`` file for a list of the requirements.

2. Install a C compiler which supports OpenMP
---------------------------------------------

Linux
^^^^^

* gcc (https://gcc.gnu.org) should be already installed, so no action is required.


Mac OS X
^^^^^^^^

* Installations of Xcode on Mac OS X come with the LLVM (clang) compiler, but it does not currently support OpenMP, so you must install gcc (https://gcc.gnu.org). This is easily done by installing the Homebrew package manager (http://brew.sh) and running:

.. code-block:: none

    $ brew install gcc-6 --without-multilib

Microsoft Windows
^^^^^^^^^^^^^^^^^

* Download and install Microsoft Visual C++ Build Tools 2015 directly from http://go.microsoft.com/fwlink/?LinkId=691126. Use the default installation options.

You can also download Microsoft Visual C++ Build Tools 2015 by going to https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx and choosing Visual Studio Downloads -> Tools for Visual Studio 2015 -> Microsoft Visual C++ Build Tools 2015.

3. Build and install gprMax
---------------------------

Once you have installed the aforementioned tools follow these steps to build and install gprMax:

* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/Mac OS X) or :code:`activate gprMax` (Windows). Run the following commands:

.. code-block:: none

    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

**You are now ready to proceed to running gprMax.**


Running gprMax
==============

gprMax in designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory.

Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/Mac OS X) or :code:`activate gprMax` (Windows)

Basic usage of gprMax is:

.. code-block:: none

    (gprMax)$ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: none

    (gprMax)$ python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out

Your results should like those from the A-scan from a metal cylinder example in introductory/basic 2D models section (http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results).

When you are finished using gprMax, the conda environment can be deactivated using :code:`source deactivate` (Linux/Mac OS X)  or :code:`deactivate` (Windows).

Optional command line arguments
-------------------------------

There are optional command line arguments for gprMax:

* ``-n`` is used along with a integer number to specify the number of times to run the input file. This option can be used to run a series of models, e.g. to create a B-scan.
* ``-mpi`` is a flag to turn on Message Passing Interface (MPI) task farm functionality. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using MPI. For further details see the Parallel performance section (http://docs.gprmax.com/en/latest/openmp_mpi.html)
* ``-benchmark`` is a flag to turn on benchmarking mode. This can be used to benchmark the threading (parallel) performance of gprMax on different hardware. For further details see the benchmarking section (http://docs.gprmax.com/en/latest/benchmarking.html)
* ``--geometry-only`` will build a model and produce any geometry views but will not run the simulation. This option is useful for checking the geometry of the model is correct.
* ``--geometry-fixed`` can be used when running a series of models where the geometry does not change between runs, e.g. a B-scan where only sources and receivers, moved using ``#src_steps`` and ``#rx_steps``, change from run to run.
* ``--opt-taguchi`` will run a series of simulations using a optimisation process based on Taguchi's method. For further details see the user libraries section (http://docs.gprmax.com/en/latest/user_libs_opt_taguchi.html)
* ``--write-processed`` will write an input file after any Python code and include commands in the original input file have been processed.
* ``-h`` or ``--help`` can be used to get help on command line options.

For example, to check the geometry of a model:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/heterogeneous_soil.in --geometry-only

For example, to run a B-scan with 60 traces:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60


Updating gprMax
===============

* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/Mac OS X) or :code:`activate gprMax` (Windows). Run the following commands:

.. code-block:: none

    (gprMax)$ git pull
    (gprMax)$ python setup.py cleanall
    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

This will pull the most recent gprMax source code form GitHub, remove/clean previously built modules, and then build and install the latest version of gprMax.


Updating conda and Python packages
----------------------------------

Periodly you should update conda and the required Python packages. To update conda, with the gprMax environment deactivated, run the following command:

.. code-block:: none

    $ conda update conda

Then you can update all the packages that are part of the gprMax environment by activating the gprMax environment and running the following command:

.. code-block:: none

    (gprMax) $ conda env update -f conda_env.yml

