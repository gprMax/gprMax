
***************
Getting Started
***************

What is gprMax?
===============

gprMax (http://www.gprmax.com) is free software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is released under the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).

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
        docs/
        gprMax/
        LICENSE
        README.rst
        setup.py
        tests/
        tools/
        user_libs/
        user_models/


* ``conda_env.yml`` is a configuration file for Anaconda (Miniconda) that sets up a Python environment with all the required Python packages for gprMax.
* ``docs`` contains source files for the User Guide. The User Guide is written using reStructuredText (http://docutils.sourceforge.net/rst.html) markup, and is built using Sphinx (http://sphinx-doc.org) and Read the Docs (https://readthedocs.org).
* ``gprMax`` is the main package. Within this package the main module is ``gprMax.py``
* ``LICENSE`` contains information on the GNU General Public License v3 or higher (http://www.gnu.org/copyleft/gpl.html).
* ``README.rst`` contains getting started information on installation, usage, and new features/changes.
* ``setup.py`` is used to compile the Cython extension modules.
* ``tests`` is a sub-package which contains test modules and input files.
* ``tools`` is a sub-package which contains scripts to assist with viewing and post-processing output from models.
* ``user_libs`` is a sub-package where useful modules contributed by users are stored.
* ``user_models`` is a sub-package where useful input files contributed by users are stored.

Installation
============

You should use the following guidance to install gprMax if you are an end-user (i.e. you don't intend to develop or contribute to the software). Developers (or those intending to use gprMax in a HPC environment) should follow the Installation for developers section (http://docs.gprmax.com/en/latest/includereadme.html#installation-for-developers).

The steps are:

1. Get the code
2. Install Python and required Python packages
3. (*Microsoft Windows only*) Install C libraries

1. Get the code
---------------

* Download the code from https://github.com/gprMax/gprMax

    * Click on *Releases* from the top header and choose the release you want (latest is at the top).
    * Download zip files of the *source code* and *binary extensions* for your platform, ``windows-32bit`` for 32-bit or ``windows-64bit`` for 64-bit versions of Microsoft Windows, ``linux-64bit`` for 64-bit versions of Linux, or ``macosx-64bit`` for 64-bit versions of Mac OS X.
    * Expand both zip files and copy the contents of the ``windows-32bit``, ``windows-64bit``, ``linux-64bit`` or ``macosx-64bit`` directory into the ``gprMax-v.X.Y.Z/gprMax`` directory.

2. Install Python and required Python packages
----------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis.

* Install the Python 3.5 version of Miniconda for your platform from http://conda.pydata.org/miniconda.html (You can get help with installing Miniconda from http://conda.pydata.org/docs/install/quick.html)
* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and navigate into the top-level gprMax directory.
* Update conda :code:`conda update conda`
* Create an environment (using the supplied ``conda_env.yml`` environment file) for gprMax with all the necessary Python packages :code:`conda env create -f conda_env.yml`
* Activate the new environment :code:`source activate gprMax` (Linux/Mac OS X) or :code:`activate gprMax` (Windows).

.. note::
    * When you are finished using gprMax the Miniconda environment can be deactivated using :code:`source deactivate` (Linux/Mac OS X)  or :code:`deactivate` (Windows).
    * If you want to install Python and the required Python packages manually, i.e. without using Anaconda/Miniconda, look in the ``conda_env.yml`` file for a list of the requirements.

3. (*Microsoft Windows only*) Install C libraries
-------------------------------------------------

* Install the Microsoft Visual Studio 2015 C++ Redistributable (``vc_redist.x86.exe`` for 32-bit or ``vc_redist.x64.exe`` for 64-bit) from https://www.microsoft.com/en-us/download/details.aspx?id=48145.

**You are now ready to proceed to running gprMax.**

Running gprMax
==============

* Open a Terminal (Linux/Mac OS X) or Command Prompt (Windows) and navigate into the top-level gprMax directory.
* If it is not already active, activate the gprMax Miniconda environment (Linux/Mac OS X) :code:`source activate gprMax` or (Windows) :code:`activate gprMax`
* gprMax in designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory. Basic usage is:

.. code-block:: none

    python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: none

    python -m gprMax user_models/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: none

    python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out

Your results should like those from the A-scan from a metal cylinder example in introductory/basic 2D models section (http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results).

Optional command line arguments
-------------------------------

There are optional command line arguments for gprMax:

* ``--geometry-only`` will build a model and produce any geometry views but will not run the simulation. This option is useful for checking the geometry of the model is correct.
* ``-n`` is used along with a integer number to specify the number of times to run the input file. This option can be used to run a series of models, e.g. to create a B-scan that uses an antenna model.
* ``-mpi`` is a flag to turn on Message Passing Interface (MPI) task farm functionality. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using MPI. For further details see the Parallel performance section (http://docs.gprmax.com/en/latest/openmp_mpi.html#parallelism-openmp-mpi).
* ``--write-python`` will write an input file after any Python code blocks in the original input file have been processed.
* ``-h`` or ``--help`` can be used to get help on command line options.

For example, to check the geometry of a model:

.. code-block:: none

    python -m gprMax user_models/heterogeneous_soil.in --geometry-only

For example, to run a B-scan with 60 traces:

.. code-block:: none

    python -m gprMax user_models/cylinder_Bscan_2D.in -n 60




