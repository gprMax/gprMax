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


MPI + OpenMP
============

There are two ways to use MPI with gprMax:

- Domain decomposition - divides a single model is across multiple MPI ranks.
- Task farm - distribute multiple models as independent tasks to each MPI rank.

.. _mpi_domain_decomposition:

MPI domain decomposition example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of a job script for running a model across multiple tasks in an HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../toolboxes/Utilities/HPC/gprmax_omp_mpi.sh
    :language: bash
    :linenos:

In this example, the model will be divided across 8 MPI ranks in a 2 x 2 x 2 pattern:

.. figure:: ../../images_shared/mpi_domain_decomposition.png
    :width: 80%
    :align: center
    :alt: MPI domain decomposition diagram

    The full model (left) is evenly divided across MPI ranks (right).

The ``--mpi`` argument is passed to gprMax which takes three integers to define the number of MPI processes in the x, y, and z dimensions to form a cartesian grid.

The ``NSLOTS`` variable which is required to set the total number of slots/cores for the parallel environment ``-pe mpi`` is usually the number of MPI tasks multiplied by the number of OpenMP threads per task. In this example the number of MPI tasks is 8 and the number of OpenMP threads per task is 16, so 128 slots are required.

Decomposition of Fractal Geometry
---------------------------------

There are some restrictions when using MPI domain decomposition with
:ref:`fractal user objects <fractals>`.

.. warning::

    gprMax will throw an error during the model build phase if the MPI
    decomposition is incompatible with the model geometry.

**#fractal_box**

When a ``#fractal_box`` has a mixing model attached, it will perform a
parallel fast Fourier transforms (FFTs) as part of its construction. To
support this, the MPI domain decomposition of the fractal box must have
size one in at least one dimension:

.. _fractal_domain_decomposition:
.. figure:: ../../images_shared/fractal_domain_decomposition.png

    Example slab and pencil decompositions. These decompositions could
    be specified with ``--mpi 8 1 1`` and ``--mpi 3 3 1`` respectively.

.. note::

    This does not necessarily mean the whole model domain needs to be
    divided this way. So long as the volume covered by the fractal box
    is divided into either slabs or pencils, the model can be built.
    This includes the volume covered by attached surfaces added by the
    ``#add_surface_water``, ``#add_surface_roughness``, or
    ``#add_grass`` commands.

**#add_surface_roughness**

When adding surface roughness, a parallel fast Fourier transform is
applied across the 2D surface of a fractal box. Therefore, the MPI
domain decomposition across the surface must be size one in at least one
dimension.

For example, in figure :numref:`fractal_domain_decomposition`, surface
roughness can be attached to any surface when using the slab
decomposition. However, if using the pencil decomposition, it could not
be attached to the XY surfaces.

**#add_grass**

Domain decomposition of grass is not currently supported. Grass can
still be built in a model so long as it is fully contained within a
single MPI rank.

MPI task farm example
^^^^^^^^^^^^^^^^^^^^^

:download:`gprmax_omp_taskfarm.sh <../../toolboxes/Utilities/HPC/gprmax_omp_taskfarm.sh>`

Here is an example of a job script for running models, e.g. A-scans to make a B-scan, distributed as independent tasks in an HPC environment using MPI. The behaviour of most of the variables is explained in the comments in the script.

.. literalinclude:: ../../toolboxes/Utilities/HPC/gprmax_omp_taskfarm.sh
    :language: bash
    :linenos:

In this example, 10 models will be distributed as independent tasks in an HPC environment using MPI.

The ``--taskfarm`` argument is passed to gprMax which takes the number of MPI tasks to run. This should be the number of models (worker tasks) plus one extra for the master task.

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
