.. _openmp-mpi:

************************
Parallelism - OpenMP/MPI
************************

OpenMP
======

The most computationally intensive parts of gprMax, which are the FDTD solver loops, have been parallelised using OpenMP (http://openmp.org) which supports multi-platform shared memory multiprocessing.

By default gprMax will try to lookup and use the maximum number of OpenMP threads (usually the number of CPU cores) available on your machine. You can override this behaviour in two ways: firstly, gprMax will check to see if the ``#num_threads`` command is present in your input file; if not, gprMax will check to see if the environment variable ``OMP_NUM_THREADS`` is set. This can be useful if you are running gprMax on a cluster or in a HPC environment where you might not want to use all of the available CPU cores.

MPI
===

The Message Passing Interface (MPI) has been utilised to implement a simple task farm that can be used to distribute a series of models as independent tasks. This can be useful in many GPR simulations where a B-scan (composed of multiple A-scans) is required. Each A-scan can be task-farmed as a independent model. Within each independent model OpenMP threading will continue to be used. Overall this creates what is know as a mixed mode OpenMP/MPI job.

By default the MPI task farm functionality is turned off. It can be switched on using the ``-mpi`` command line flag. MPI requires an installation of OpenMPI (http://www.open-mpi.org) and the mpi4py Python package.

.. note::

        It seems due to a lack of interest there are no binary versions of OpenMPI available for Microsoft Windows (https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php)

Running gprMax using the MPI task farm functionality is heavily dependent on the configuration of your machine/cluster. The following example is intended as general guidance to help you get started.

Grid Engine example
-------------------

Clusters usually requires jobs to be submitted to a queue using a job script. Typically within that script the ``mpirun`` program is used to execute MPI jobs. Here is an example of a job script for a cluster that uses Oracle (Sun) Grid Engine. The behaviour of most of the variables is explained in the comments in the script.

.. code-block:: none

    #!/bin/bash
    #####################################################################################
    ### Specify bash shell:
    #$ -S /bin/bash

    ### Change to current working directory:
    #$ -cwd

    ### Specify runtime (hh:mm:ss):
    #$ -l h_rt=01:00:00

    ### Email options:
    #$ -m ea -M joe.bloggs@email.com

    ### Parallel environment ($NSLOTS):
    #$ -pe openmpi_fillup_mark2 80

    ### Job script name:
    #$ -N test_mpi.sh
    #####################################################################################

    ### Initialise environment module
    . /etc/profile.d/modules.sh

    ### Load Anaconda environment with Python 3 and packages
    module load anaconda
    source activate python3

    ### Load OpenMPI
    module load openmpi-gcc

    ### Set number of OpenMP threads
    export OMP_NUM_THREADS=8

    ### Run gprMax with input file
    cd $HOME/gprMax
    mpirun -np $NSLOTS python -m gprMax test.in -n 10 -mpi

The ``NSLOTS`` variable is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 10 and number of OpenMP threads per task is 8, so 8O slots are required.


