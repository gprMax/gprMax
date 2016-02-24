.. _benchmarking:

************
Benchmarking
************

This section provides information and results from performance benchmarking of gprMax.

How to benchmark?
=================

The following simple model is an example (found in the ``tests/benchmarking`` sub-package) that can be used to benchmark gprMax on your own system. The model contains a simple source in free space.

.. literalinclude:: ../../tests/benchmarking/bench_100x100x100.in
    :language: none
    :linenos:

The ``#num_threads`` command should be adjusted from 1 up to the number of physical CPU cores on your machine, the model run, and the solving time recorded.


Results
=======

Mac OS X
--------

iMac (Retina 5K, 27-inch, Late 2014), Mac OS X 10.11.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../tests/benchmarking/results/MacOSX/Darwin-15.3.0-x86_64-i386-64bit.png
    :width: 600px

    Execution time and speed-up factor plots for gprMax (v3b21) and GprMax (v2).

The results demonstrate that the new (v3) code written in Python and Cython is faster, in these two benchmarks, that the old (v2) code which was written in C. It also shows that the performance scaling with multiple OpenMP threads is better with the old (v2) code.

Zero threads signifies that the code was compiled serially, i.e. without using OpenMP. Results from the old (v2) code show that when it is compiled serially the performance is approximately the same as when it is compiled with OpenMP and run with a single thread. With the new (v3) code this is not the case. The overhead in setting up and tearing down the OpenMP threads means that for a single thread the performance is worse than the serially-compiled version.