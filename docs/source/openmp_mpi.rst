.. _openmp-mpi:

***********
Parallelism
***********

OpenMP
======

The most computationally intensive parts of gprMax, which are the FDTD solver loops, have been parallelised using OpenMP (http://openmp.org) which supports multi-platform shared memory multiprocessing.

By default gprMax will try to determine and use the maximum number of OpenMP threads (usually the number of physical CPU cores) available on your machine. You can override this behaviour in two ways: firstly, gprMax will check to see if the ``#num_threads`` command is present in your input file; if not, gprMax will check to see if the environment variable ``OMP_NUM_THREADS`` is set. This can be useful if you are running gprMax in a High-Performance Computing (HPC) environment where you might not want to use all of the available CPU cores.

MPI
===

The Message Passing Interface (MPI) has been utilised to implement a simple task farm that can be used to distribute a series of models as independent tasks. This can be useful in many GPR simulations where a B-scan (composed of multiple A-scans) is required. Each A-scan can be task-farmed as a independent model. Within each independent model OpenMP threading will continue to be used (as described above). Overall this creates what is know as a mixed mode OpenMP/MPI job.

By default the MPI task farm functionality is turned off. It can be switched on using the ``-mpi`` command line flag. MPI requires an installation of the ``mpi4py`` Python package, which itself depends on an underlying MPI installation, usually OpenMPI (http://www.open-mpi.org). On Microsoft Windows ``mpi4py`` requires Microsoft MPI 6 (https://www.microsoft.com/en-us/download/details.aspx?id=47259).

HPC job scripts
===============

HPC environments usually require jobs to be submitted to a queue using a job script. The following are examples of job scripts for a HPC environment that uses Oracle (Sun) Grid Engine, and are intended as general guidance to help you get started. Using gprMax in an HPC environment is heavily dependent on the configuration of your specific HPC/cluster, e.g. the names of parallel environments (``-pe``) and compiler modules will depend on how they were defined by your system administrator.

OpenMP example
--------------

:download:`gprmax_openmp.sh <../../tools/HPC scripts/gprmax_omp.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, one after another on a single cluster node. This is not as beneficial as the OpenMP/MPI example, but it can be a helpful starting point when getting the software running in your HPC environment. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../tools/HPC scripts/gprmax_omp.sh
    :language: bash
    :linenos:

In this example 100 models will be run one after another on a single node of the cluster. Each model will be parallelised using 8 OpenMP threads.


OpenMP/MPI example
------------------

:download:`gprmax_openmp.sh <../../tools/HPC scripts/gprmax_omp_mpi.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, distributed as independent tasks in a HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../tools/HPC scripts/gprmax_omp_mpi.sh
    :language: bash
    :linenos:

In this example 100 models will be distributed as independent tasks in a HPC environment using MPI. The ``NSLOTS`` variable is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 100 and number of OpenMP threads per task is 8, so 800 slots are required.

.. tip::
    These example scripts can be used directly on Eddie, the Edinburgh Compute and Data Facility (ECDF) - http://www.ed.ac.uk/information-services/research-support/research-computing/ecdf/high-performance-computing


