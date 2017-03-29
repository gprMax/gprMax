.. image:: https://readthedocs.org/projects/gprmax/badge/?version=latest
    :target: http://docs.gprmax.com/en/latest/?badge=latest
    :alt: Documentation Status
|
.. image:: /docs/source/images/gprMax_logo_small.png
    :target: http://www.gprmax.com

***************
Getting Started
***************

What is gprMax?
===============

`gprMax <http://www.gprmax.com>`_ is open source software that simulates electromagnetic wave propagation. It solves Maxwell's equations in 3D using the Finite-Difference Time-Domain (FDTD) method. gprMax was designed for modelling Ground Penetrating Radar (GPR) but can also be used to model electromagnetic wave propagation for many other applications.

gprMax is currently released under the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.

gprMax is written in `Python <https://www.python.org>`_ 3 and includes performance-critical parts written in `Cython <http://cython.org>`_ with `OpenMP <http://www.openmp.org>`_.

Using gprMax? Cite us
---------------------

If you use gprMax and publish your work we would be grateful if you could cite our work using:

* Warren, C., Giannopoulos, A., & Giannakis I. (2016). gprMax: Open source software to simulate electromagnetic wave propagation for Ground Penetrating Radar, `Computer Physics Communications` (http://dx.doi.org/10.1016/j.cpc.2016.08.020)

For further information on referencing gprMax visit the `Publications section of our website <http://www.gprmax.com/publications.shtml>`_.


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
* ``docs`` contains source files for the User Guide. The User Guide is written using `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ markup, and is built using `Sphinx <http://sphinx-doc.org>`_ and `Read the Docs <https://readthedocs.org>`_.
* ``gprMax`` is the main package. Within this package the main module is ``gprMax.py``
* ``LICENSE`` contains information on the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_.
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

You can `watch screencasts <http://docs.gprmax.com/en/latest/screencasts.html>`_ that demonstrate the installation and update processes.

1. Install Python, required Python packages, and get gprMax source
------------------------------------------------------------------

We recommend using Miniconda to install Python and the required Python packages for gprMax in a self-contained Python environment. Miniconda is a mini version of Anaconda which is a completely free Python distribution (including for commercial use and redistribution). It includes more than 300 of the most popular Python packages for science, math, engineering, and data analysis.

* `Download and install Miniconda <http://conda.pydata.org/miniconda.html>`_. Choose the Python 3.6 version for your platform (see the `Quick Install page <http://conda.pydata.org/docs/install/quick.html>`_ for help installing Miniconda)
* Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:

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

* `gcc <https://gcc.gnu.org>`_ should be already installed, so no action is required.


macOS
^^^^^

* Xcode (the IDE for macOS) comes with the LLVM (clang) compiler, but it does not currently support OpenMP, so you must install `gcc <https://gcc.gnu.org>`_. This can be done by downloading and installing the `Homebrew package manager <http://brew.sh>`_ and running:

.. code-block:: none

    $ brew install gcc --without-multilib

Microsoft Windows
^^^^^^^^^^^^^^^^^

* Download and install Build Tools for Visual Studio 2017 from the `Visual Studio downloads page <https://www.visualstudio.com/downloads/>`_ in the section Other Tools and Frameworks. Use the default installation options.

3. Build and install gprMax
---------------------------

Once you have installed the aforementioned tools follow these steps to build and install gprMax:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/macOS) or :code:`activate gprMax` (Windows). Run the following commands:

.. code-block:: none

    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

**You are now ready to proceed to running gprMax.**


Running gprMax
==============

gprMax in designed as a Python package, i.e. a namespace which can contain multiple packages and modules, much like a directory.

Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/macOS) or :code:`activate gprMax` (Windows)

Basic usage of gprMax is:

.. code-block:: none

    (gprMax)$ python -m gprMax path_to/name_of_input_file

For example to run one of the test models:

.. code-block:: none

    (gprMax)$ python -m gprMax user_models/cylinder_Ascan_2D.in

When the simulation is complete you can plot the A-scan using:

.. code-block:: none

    (gprMax)$ python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out

Your results should like those from the A-scan from the metal cylinder example in `introductory/basic 2D models section <http://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results>`_

When you are finished using gprMax, the conda environment can be deactivated using :code:`source deactivate` (Linux/macOS)  or :code:`deactivate` (Windows).

Optional command line arguments
-------------------------------

====================== ======= ===========
Argument name          Type    Description
====================== ======= ===========
``-n``                 integer number of times to run the input file. This option can be used to run a series of models, e.g. to create a B-scan with 60 traces: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60``
``-restart``           integer model number to start/restart simulation from. It would typically be used to restart a series of models from a specific model number, with the ``-n`` argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 15 -restart 45``
``-task``              integer task identifier (model number) when running simulation as a job array on `Open Grid Scheduler/Grid Engine <http://gridscheduler.sourceforge.net/index.html>`_. For further details see the `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
``-mpi``               integer number of Message Passing Interface (MPI) tasks, i.e. master + workers, for MPI task farm. This option is most usefully combined with ``-n`` to allow individual models to be farmed out using a MPI task farm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace: ``(gprMax)$ python -m gprMax user_models/cylinder_Bscan_2D.in -n 60 -mpi 61``. For further details see the `parallel performance section of the User Guide <http://docs.gprmax.com/en/latest/openmp_mpi.html>`_
``-benchmark``         flag    switch on benchmarking mode. This can be used to benchmark the threading (parallel) performance of gprMax on different hardware. For further details see the `benchmarking section of the User Guide <http://docs.gprmax.com/en/latest/benchmarking.html>`_
``--geometry-only``    flag    build a model and produce any geometry views but do not run the simulation, e.g. to check the geometry of a model is correct: ``(gprMax)$ python -m gprMax user_models/heterogeneous_soil.in --geometry-only``
``--geometry-fixed``   flag    run a series of models where the geometry does not change between models, e.g. a B-scan where *only* the position of simple sources and receivers, moved using ``#src_steps`` and ``#rx_steps``, changes between models.
``--opt-taguchi``      flag    run a series of models using an optimisation process based on Taguchi's method. For further details see the `user libraries section of the User Guide <http://docs.gprmax.com/en/latest/user_libs_opt_taguchi.html>`_
``--write-processed``  flag    write another input file after any Python code and include commands in the original input file have been processed. Useful for checking that any Python code is being correctly processed into gprMax commands.
``-h`` or ``--help``   flag    used to get help on command line options.
====================== ======= ===========

Updating gprMax
===============

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level gprMax directory, and if it is not already active, activate the gprMax conda environment :code:`source activate gprMax` (Linux/macOS) or :code:`activate gprMax` (Windows). Run the following commands:

.. code-block:: none

    (gprMax)$ git pull
    (gprMax)$ python setup.py cleanall
    (gprMax)$ python setup.py build
    (gprMax)$ python setup.py install

This will pull the most recent gprMax source code form GitHub, remove/clean previously built modules, and then build and install the latest version of gprMax.


Updating conda and Python packages
----------------------------------

Periodically you should update conda and the required Python packages. With the gprMax environment deactivated and from the top-level gprMax directory, run the following commands:

.. code-block:: none

    $ conda update conda
    $ conda env update -f conda_env.yml

