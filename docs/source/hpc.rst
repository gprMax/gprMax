.. _hpc:

***
HPC
***

High-performance computing (HPC) environments usually require jobs to be submitted to a queue using a job script. The following are examples of job scripts for an HPC environment that uses `Open Grid Scheduler/Grid Engine <http://gridscheduler.sourceforge.net/index.html>`_, and are intended as general guidance to help you get started. Using gprMax in an HPC environment is heavily dependent on the configuration of your specific HPC/cluster, e.g. the names of parallel environments (``-pe``) and compiler modules will depend on how they were defined by your system administrator.


OpenMP example
==============

:download:`gprmax_omp.sh <../../toolboxes/Utilities/HPC/gprmax_omp.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, one after another on a single cluster node. This is not as beneficial as the OpenMP/MPI example, but it can be a helpful starting point when getting the software running in your HPC environment. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../toolboxes/Utilities/HPC/gprmax_omp.sh
    :language: bash
    :linenos:

In this example 10 models will be run one after another on a single node of the cluster (on this particular cluster a single node has 16 cores/threads available). Each model will be parallelised using 16 OpenMP threads.


OpenMP/MPI example
==================

:download:`gprmax_omp_mpi.sh <../../toolboxes/Utilities/HPC/gprmax_omp_mpi.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, distributed as independent tasks in an HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../toolboxes/Utilities/HPC/gprmax_omp_mpi.sh
    :language: bash
    :linenos:

In this example, 10 models will be distributed as independent tasks in an HPC environment using MPI.

The ``-mpi`` argument is passed to gprMax which takes the number of MPI tasks to run. This should be the number of models (worker tasks) plus one extra for the master task.

The ``NSLOTS`` variable which is required to set the total number of slots/cores for the parallel environment ``-pe mpi`` is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 11 and the number of OpenMP threads per task is 16, so 176 slots are required.


Job array example
=================

:download:`gprmax_omp_jobarray.sh <../../toolboxes/Utilities/HPC/gprmax_omp_jobarray.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, using the job array functionality of Open Grid Scheduler/Grid Engine. A job array is a single submit script that is run multiple times. It has similar functionality, for gprMax, to using the aforementioned MPI task farm. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../toolboxes/Utilities/HPC/gprmax_omp_jobarray.sh
    :language: bash
    :linenos:

The ``-t`` tells Grid Engine that we are using a job array followed by a range of integers which will be the IDs for each individual task (model). Task IDs must start from 1, and the total number of tasks in the range should correspond to the number of models you want to run, i.e. the integer with the ``-n`` flag passed to gprMax. The ``-i`` flag is passed to gprMax along with the specific number of the task (model) with the environment variable ``$SGE_TASK_ID``.

A job array means that exactly the same submit script is going to be run multiple times, the only difference between each run is the environment variable ``$SGE_TASK_ID``.
