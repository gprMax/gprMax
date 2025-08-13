.. _benchmarking:

************
Benchmarking
************

This section provides information and results from performance benchmarking of gprMax.

How to benchmark?
=================

The following simple models (found in the ``testing/benchmarking`` sub-package) can be used to benchmark gprMax on your own system. The models feature different domain sizes (from 100^3 to 800^3 cells) and contain a simple Hertzian dipole source in free space:

.. literalinclude:: ../../testing/benchmarking/bench_simple.py
    :language: python
    :linenos:

The performance metric used to measure the throughput of the solver is:

.. math::

    P = \frac{NX \cdot NY \cdot NZ \cdot NT}{T \cdot 1 \times 10^6},

where P is the throughput in millions of cells per second; NX, NY, and NZ are the number of cells in domain in the x, y, and z directions; NT is the number of time-steps in the simulation; and T is the runtime of the simulation in seconds.

Apple Metal GPU Benchmarking
=============================

For macOS users with Apple Silicon (M-series) based GPUs, a dedicated Metal benchmarking script is available in the ``testing/benchmarking`` sub-package:

.. literalinclude:: ../../testing/benchmarking/benchmark_metal.py
    :language: python
    :linenos:
    :lines: 1-30

This script provides comprehensive benchmarking capabilities specifically designed for the Apple Metal backend:

Features
--------

* **Automated domain size testing**: Tests multiple domain sizes from 50×50×50 to 200×200×200 cells
* **CPU vs Metal comparison**: Runs identical simulations on both CPU and Metal backends for direct performance comparison
* **Performance visualization**: Generates plots showing throughput (Mcells/s) and speedup ratios
* **Data export**: Saves results in multiple formats (JSON, NumPy) for further analysis
* **Validation integration**: Can be combined with PML validation testing

Usage
-----

To run the Metal benchmarking suite:

.. code-block:: none

    (gprMax)$ cd testing/benchmarking
    (gprMax)$ python benchmark_metal.py

The script will automatically:

1. Create benchmark input files for different domain sizes
2. Run simulations using both CPU and Metal backends
3. Calculate performance metrics using the standard formula above
4. Generate comparison plots and save results

Results
-------

The Apple Metal backend has been benchmarked extensively and shows:

* **Peak performance**: 849.5 Mcells/s achieved with 200×200×200 cell domains
* **Optimal speedup**: Up to 1.27× improvement over CPU execution for large domains
* **Memory efficiency**: Leverages unified memory architecture for reduced overhead
* **Scaling characteristics**: Performance improvements increase with domain size

Visualization Tools
===================

Additional plotting utilities are available for advanced benchmarking analysis:

.. literalinclude:: ../../testing/benchmarking/plot_gpu_benchmark.py
    :language: python
    :linenos:
    :lines: 1-25

This plotting script enables:

* **Multi-platform comparison**: Compare results across different hardware configurations
* **Custom data visualization**: Load and plot benchmark data from various sources
* **Performance trend analysis**: Visualize performance scaling with domain size
* **Publication-ready plots**: Generate high-quality figures for reports and papers

The script can load data from the Metal benchmarking results and create comparative plots showing the performance characteristics of Apple Metal against other accelerators.

.. figure:: ../../images_shared/GPU_NVIDIA.png
    :width: 600px
