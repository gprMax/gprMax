.. _hpc:

***
HPC
***

Using gprMax in an HPC environment depends on the configuration of your
cluster: compiler and library modules, programming environments, and job
submission processes vary between systems.

.. note::

    General details about the types of acceleration available in gprMax are shown in the :ref:`accelerators` section.


Installation
============

Follow the :ref:`installation guide <installation>` to choose the package or
source route and preserve any existing v3 environment. A cluster may supply
Python, compilers and native libraries through modules instead of Conda.

For example, the following is an illustrative source-build recipe using the
`ARCHER2 <https://www.archer2.ac.uk/>`_ module environment. Obtain a separate
v4 checkout following :ref:`installation-source`, select the desired revision,
and run these commands from its root. Check your site's module versions and
the supported Python versions in the installation guide; do not assume that
an unqualified clone of the default branch is the intended v4 revision:

.. code-block:: console

    $ module load PrgEnv-gnu
    $ module load cray-python
    $ module load cray-fftw
    $ module load cray-hdf5-parallel
    $ export CC=cc
    $ export CXX=CC
    $ export FC=ftn
    $ python -m venv --system-site-packages --prompt gprMax .venv
    $ source .venv/bin/activate
    (gprMax)$ python -m pip install --upgrade pip
    (gprMax)$ python -m pip install -r requirements.txt
    (gprMax)$ HDF5_MPI='ON' python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=h5py h5py
    (gprMax)$ python -m pip install -e ".[mpi]"
    (gprMax)$ python -c "import h5py; print(h5py.get_config().mpi)"

Here ``--system-site-packages`` intentionally exposes packages supplied by
the loaded HPC Python module; omit it when that access is not wanted. This
is a site-specific choice, not a requirement for ordinary PyPI installation.
The forced h5py rebuild avoids silently retaining a serial installation;
the final check must print ``True`` for parallel HDF5 output.

.. tip::

    Consult your system's documentation for site specific information.

Job Submission examples
=======================

High-performance computing (HPC) environments usually require jobs to be submitted to a queue using a job script. The following are examples of job scripts for an HPC environment that uses `Open Grid Scheduler/Grid Engine <http://gridscheduler.sourceforge.net/index.html>`_, and are intended as general guidance to help you get started. The names of parallel environments (``-pe``) and compiler modules will depend on how they were defined by your system administrator.

The Grid Engine examples below use Bash and activate ``gprMax-v4`` through a
site-provided Anaconda module. Adapt that setup to the interpreter/environment
you installed; Conda is not required. Submit from the working directory that
contains ``mymodel.in``. The ``-cwd`` directive keeps that directory, so the
scripts do not change into a possibly unrelated old gprMax checkout.

OpenMP
^^^^^^

:download:`gprmax_omp.sh <../../gprMax/toolboxes/Utilities/HPC/gprmax_omp.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, one after another on a single cluster node. This is not as beneficial as the OpenMP/MPI example, but it can be a helpful starting point when getting the software running in your HPC environment. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../gprMax/toolboxes/Utilities/HPC/gprmax_omp.sh
    :language: bash
    :linenos:

In this example 10 models will be run one after another on a single node of the cluster (on this particular cluster a single node has 16 cores/threads available). Each model will be parallelised using 16 OpenMP threads.

MPI domain decomposition
^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of a job script for running a model across multiple tasks in an HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. note::

    This example is based on the `ARCHER2 <https://www.archer2.ac.uk/>`_ system and uses the `SLURM <https://slurm.schedmd.com/>`_ scheduler.

.. literalinclude:: ../../gprMax/toolboxes/Utilities/HPC/gprmax_omp_mpi.sh
    :language: bash
    :linenos:

In this example, the model will be divided across 8 MPI ranks in a 2 x 2 x 2 pattern:

.. figure:: ../../images_shared/mpi_domain_decomposition.png
    :width: 80%
    :align: center
    :alt: MPI domain decomposition diagram

    The full model (left) is evenly divided across MPI ranks (right).

The ``--mpi`` argument is passed to gprMax which takes three integers to define the number of MPI processes in the x, y, and z dimensions to form a cartesian grid.

Point sources and receivers have a single MPI owner. An internal partition
plane belongs to the rank on its positive side; a global upper plane belongs
to the last rank along that axis, including on PMC symmetry faces. This
ownership rule does not relax source-specific placement restrictions or make
every field component valid on a boundary.

Unlike the grid engine examples, here we specify the number of CPUs per task (16) and the number of tasks (8), rather than the total number of CPUs/slots.

.. note::

    Some restrictions apply to the domain decomposition when using fractal geometry as explained :ref:`here <fractal_domain_decomposition>`.

MPI task farm
^^^^^^^^^^^^^

:download:`gprmax_omp_taskfarm.sh <../../gprMax/toolboxes/Utilities/HPC/gprmax_omp_taskfarm.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, distributed as independent tasks in an HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../gprMax/toolboxes/Utilities/HPC/gprmax_omp_taskfarm.sh
    :language: bash
    :linenos:

In this example, 10 models will be distributed as independent tasks in an HPC environment using MPI.

``--taskfarm`` is a boolean switch; it does not take a task count. The MPI
launcher sets the number of ranks, while ``-n`` sets the number of models.
One rank coordinates the farm and the remaining ranks execute complete models.
Workers can each process several models, so a worker per model is not required.
For example, ``mpiexec -n 3 python -m gprMax model.in -n 10 --taskfarm`` runs
ten models using two workers and one coordinator.

If a worker's model raises an exception, other submitted models are still
processed. After all jobs finish, the coordinator raises ``TaskfarmError``
and the command exits unsuccessfully; successful output files are retained.
The exception's ``results`` and ``failures`` attributes expose the outcomes
to Python callers. Distributed-domain failures likewise use a nonzero MPI
abort status rather than reporting a successful run.

The ``NSLOTS`` variable which is required to set the total number of slots/cores for the parallel environment ``-pe mpi`` is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 11 and the number of OpenMP threads per task is 16, so 176 slots are required.


Job array
^^^^^^^^^

:download:`gprmax_omp_jobarray.sh <../../gprMax/toolboxes/Utilities/HPC/gprmax_omp_jobarray.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, using the job array functionality of Open Grid Scheduler/Grid Engine. A job array is a single submit script that is run multiple times. It has similar functionality, for gprMax, to using the aforementioned MPI task farm. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../gprMax/toolboxes/Utilities/HPC/gprmax_omp_jobarray.sh
    :language: bash
    :linenos:

The scheduler's ``-t 1-10`` launches ten tasks. Each task passes its
``$SGE_TASK_ID`` as the one-based starting model number, ``-i``, and uses
``-n 1`` to execute exactly one model. For ordinary runs, ``-n`` is the number
of models to execute from ``-i``, not the total size of the scheduler array.
Using ``-n 10`` in every task would launch overlapping batches.

The explicit ``-o mymodel_TASK_ID`` prefix keeps output filenames distinct.
A one-model run does not automatically append its ``-i`` value. For Python
input blocks that depend on the total survey size, supply that survey size
separately rather than increasing ``-n``. This example concerns ordinary
stepped runs, not study-managed restart rules.

A job array means that exactly the same submit script is going to be run multiple times, the only difference between each run is the environment variable ``$SGE_TASK_ID``.
