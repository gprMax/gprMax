.. _benchmarking:

************************
Performance benchmarking
************************

This section provides information and results from performance benchmarking of gprMax.

How to benchmark?
=================

The following simple models (found in the ``tests/benchmarking`` sub-package) can be used to benchmark gprMax on your own system. The models feature different domain sizes (from 100^3 to 450^3 cells) and contain a simple Hertzian dipole source in free space. The following shows an example of the 100^3 cell model:


.. literalinclude:: ../../tests/benchmarking/bench_100x100x100.in
    :language: none
    :linenos:


Using the following steps to collect and report benchmarking results for each of the models:

1. Run gprMax in benchmarking mode, e.g. ``python -m gprMax tests/benchmarking/bench_100x100x100.in -benchmark``
2. Use the ``plot_benchmark`` module to create plots of the execution time and speed-up, e.g. ``python -m tests.benchmarking.plot_benchmark tests/benchmarking/bench_100x100x100.npz``. You can combine results into a single plot, e.g. e.g. ``python -m tests.benchmarking.plot_benchmark tests/benchmarking/bench_100x100x100.npz --otherresults tests/benchmarking/bench_150x150x150.npz``.
3. Share your data by emailing us your Numpy archives and plot files to info@gprmax.com

Results: CPU
============

Mac OS X
--------

iMac15,1
^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/MacOSX/iMac15,1/Apple_iMac15,1+Ccode.png
    :width: 600px

    Execution time and speed-up factor plots for Python/Cython-based gprMax and previous (v.2) C-based code.

.. note::

    Zero threads indicates that the code was compiled serially, i.e. without using OpenMP.

The results demonstrate that the Python/Cython-based code is faster, in these two benchmarks, than the previous version which was written in C. It also shows that the performance scaling with multiple OpenMP threads is better with the C-based code. Results from the C-based code show that when it is compiled serially the performance is approximately the same as when it is compiled with OpenMP and run with a single thread. With the Python/Cython-based code this is not the case. The overhead in setting up and tearing down the OpenMP threads means that for a single thread the performance is worse than the serially-compiled version.

iMac15,1
^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/MacOSX/iMac15,1/Apple_iMac15,1.png
    :width: 600px

MacPro1,1
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/MacOSX/MacPro1,1/Apple_MacPro1,1.png
    :width: 600px


MacPro3,1
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/MacOSX/MacPro3,1/Apple_MacPro3,1.png
    :width: 600px


Linux
-----

Dell PowerEdge R630
^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/Linux/Dell_PowerEdge_R630/Dell_PowerEdge_R630.png
    :width: 600px

Lenovo System x3650 M5
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/Linux/Lenovo_System_x3650_M5/Lenovo_System_x3650_M5.png
    :width: 600px

SuperMicro SYS-7048GR-TR
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/Linux/SuperMicro_SYS-7048GR-TR/Supermicro_SYS-7048GR-TR.png
    :width: 600px


Windows
-------

Lenovo T430
^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/Windows7/Lenovo_T430/Lenovo_T430.png
    :width: 600px

Dell Z420
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/cpu/Windows7/Dell_Z420/DellZ420.png
    :width: 600px



Results: GPU
============

NVIDIA GPUs
-----------

The performance metric used to measure the throughput of the solver is:

.. math::

    P = \frac{NX \cdot NY \cdot NZ \cdot NT}{T \cdot 1 \times 10^6},

where P is the throughput in millions of cells per second; NX, NY, and NZ are the number of cells in domain in the x, y, and z directions; NT is the number of time-steps in the simulation; and T is the runtime of the simulation in seconds.

.. figure:: ../../tests/benchmarking/results/gpu/NVIDIA.png
    :width: 600px
