.. _openmp-mpi:

******************
Parallel execution
******************

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

HPC environments usually require jobs to be submitted to a queue using a job script. The following are examples of job scripts for a HPC environment that uses Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html), and are intended as general guidance to help you get started. Using gprMax in an HPC environment is heavily dependent on the configuration of your specific HPC/cluster, e.g. the names of parallel environments (``-pe``) and compiler modules will depend on how they were defined by your system administrator.


OpenMP example
--------------

:download:`gprmax_omp.sh <../../tools/HPC scripts/gprmax_omp.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, one after another on a single cluster node. This is not as beneficial as the OpenMP/MPI example, but it can be a helpful starting point when getting the software running in your HPC environment. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../tools/HPC scripts/gprmax_omp.sh
    :language: bash
    :linenos:

In this example 10 models will be run one after another on a single node of the cluster (on this particular cluster a single node has 16 cores/threads available). Each model will be parallelised using 16 OpenMP threads.


OpenMP/MPI example
------------------

:download:`gprmax_omp_mpi.sh <../../tools/HPC scripts/gprmax_omp_mpi.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, distributed as independent tasks in a HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../tools/HPC scripts/gprmax_omp_mpi.sh
    :language: bash
    :linenos:

In this example 10 models will be distributed as independent tasks in a HPC environment using MPI.

The ``-np`` flag passed to ``mpiexec`` takes the number of MPI tasks (copies of gprMax/number of models) to run. Depending on the configuration of the parallel environment (whether ``job_is_first_task`` is True/False) this will be either 10 (the number of models) or 11 (the number of models plus one extra for the master task).

The ``NSLOTS`` variable which is required to set the total number of slots/cores for the parallel environment ``-pe mpi`` is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 11 and number of OpenMP threads per task is 16, so 176 slots are required.


Job array example
-----------------

:download:`gprmax_omp_jobarray.sh <../../tools/HPC scripts/gprmax_omp_jobarray.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, using the job array functionality of Open Grid Scheduler/Grid Engine. A job array is a single submit script that is run multiple times. It has similar functionality, for gprMax, to using the aforementioned MPI task farm. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../tools/HPC scripts/gprmax_omp_jobarray.sh
    :language: bash
    :linenos:

The ``-t`` tells Grid Engine that we are using a job array followed by a range of integers which will be the IDs for each individual task (model). Task IDs must start from 1, and the total number of tasks in the range should correspond to the number of models you want to run, i.e. the integer with the ``-n`` flag passed to gprMax. The ``-taskid`` flag is given to gprMax to tell it we are using a job array, along with the specific number of the task (model) with the environment variable ``$SGE_TASK_ID``.

A job array means that exactly the same submit script is going to be run multiple times, the only difference between each run is the environment variable ``$SGE_TASK_ID``.


Eddie
-----

Eddie is the Edinburgh Compute and Data Facility (ECDF) - http://www.ed.ac.uk/information-services/research-support/research-computing/ecdf/high-performance-computing - run by the University of Edinburgh. The following are useful notes to get gprMax installed and running on eddie3 (the third iteration of the cluster):

* Git is already installed on eddie3, so you don't need to install it through Anaconda, you can proceed directly to cloning the gprMax GitHub repository with ``git clone https://github.com/gprMax/gprMax.git``

* Anaconda is already installed as an application module on eddie3. You should follow these instructions (https://www.wiki.ed.ac.uk/display/ResearchServices/Anaconda) to ensure Anaconda environments will be created in a suitable location (not your home directory as you will rapidly run out of space). Before you proceed to create the Anaconda environment for gprMax you must make sure the OpenMPI module is loaded with ``module load openmpi``. This is neccessary so that the ``mpi4py`` Python module is correctly linked to OpenMPI. You can then create the Anaconda environment with ``conda env create -f conda_env.yml``

* You should then activate the gprMax Anaconda environment, and build and install gprMax according the standard installation procedure.

* The previous job submission example scripts for OpenMP and OpenMP/MPI should run on eddie3.

* The parameter ``job_is_first_task`` is set to True for the ``mpi`` parallel environment. This means when specifying the number of MPI tasks (``-np``) you must add one to the number of copies of gprMax/number of models you wish to run.

* The ``NSLOTS`` variable for the total number of slots/cores for the parallel environment ``-pe mpi`` must be specified as a multiple of 16 (the total number of cores/threads available on a single node), e.g. 61 MPI tasks each using 4 threads would require a total 244 slots/cores. This must be rounded up to the nearest multiple of 16, e.g. 256.

