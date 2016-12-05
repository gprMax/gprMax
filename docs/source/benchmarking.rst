.. _benchmarking:

************************
Performance benchmarking
************************

This section provides information and results from performance benchmarking of gprMax.

How to benchmark?
=================

The following simple models (found in the ``tests/benchmarking`` sub-package) can be used to benchmark gprMax on your own system. The models feature different domain sizes and contain a simple source in free space.


.. literalinclude:: ../../tests/benchmarking/bench_100x100x100.in
    :language: none
    :linenos:


.. literalinclude:: ../../tests/benchmarking/bench_150x150x150.in
    :language: none
    :linenos:


.. literalinclude:: ../../tests/benchmarking/bench_200x200x200.in
    :language: none
    :linenos:


Using the following steps to collect and report benchmarking results for each of the models:

1. Run gprMax in benchmarking mode, e.g. ``python -m gprMax tests/benchmarking/bench_100x100x100.in -benchmark``
2. Use the ``plot_benchmark`` module to create plots of the execution time and speed-up, e.g. ``python -m tests.benchmarking.plot_benchmark tests/benchmarking/bench_100x100x100.npz``. You can combine results into a single plot, e.g. e.g. ``python -m tests.benchmarking.plot_benchmark tests/benchmarking/bench_100x100x100.npz --otherresults tests/benchmarking/bench_150x150x150.npz``.
3. Share your data by emailing us your Numpy archives and plot files to info@gprmax.com

Results
=======

Mac OS X
--------

iMac15,1
^^^^^^^^

.. figure:: ../../tests/benchmarking/results/MacOSX/iMac15,1/Apple_iMac15,1+Ccode.png
    :width: 600px

    Execution time and speed-up factor plots for Python/Cython-based gprMax and previous (v.2) C-based code.

.. note::

    Zero threads indicates that the code was compiled serially, i.e. without using OpenMP.

The results demonstrate that the Python/Cython-based code is faster, in these two benchmarks, than the previous version which was written in C. It also shows that the performance scaling with multiple OpenMP threads is better with the C-based code. Results from the C-based code show that when it is compiled serially the performance is approximately the same as when it is compiled with OpenMP and run with a single thread. With the Python/Cython-based code this is not the case. The overhead in setting up and tearing down the OpenMP threads means that for a single thread the performance is worse than the serially-compiled version.

iMac15,1
^^^^^^^^

.. figure:: ../../tests/benchmarking/results/MacOSX/iMac15,1/Apple_iMac15,1.png
    :width: 600px

MacPro1,1
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/MacOSX/MacPro1,1/Apple_MacPro1,1.png
    :width: 600px


MacPro3,1
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/MacOSX/MacPro3,1/Apple_MacPro3,1.png
    :width: 600px


Linux
-----

Dell PowerEdge R630
^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/Linux/Dell_PowerEdge_R630/Dell_PowerEdge_R630.png
    :width: 600px

SuperMicro SYS-7048GR-TR
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/Linux/SuperMicro_SYS-7048GR-TR/Supermicro_SYS-7048GR-TR.png
    :width: 600px


Windows
-------

Lenovo T430
^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/Windows7/Lenovo_T430/Lenovo_T430.png
    :width: 600px

Dell Z420
^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/Windows7/Dell_Z420/DellZ420.png
    :width: 600px
