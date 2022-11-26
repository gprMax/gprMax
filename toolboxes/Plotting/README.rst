Toolboxes is a sub-package where useful Python modules contributed by users are stored.

********
Plotting
********

Information
===========

This package is intended to provide some basic scripts to get started with plotting outputs from simulations.

Package contents
================

* ``plot_antenna_params.py`` plots antenna parameters - incident, reflected and total voltages and currents; s11, (s21) and input impedance from an output file containing a transmission line source.
* ``plot_Ascan.py`` plots electric and magnetic fields and currents from all receiver points in the given output file. Each receiver point is plotted in a new figure window.
* ``plot_Bscan.py`` plots a B-scan (multiple A-scans) image.
* ``plot_source_wave.py`` plot built-in waveforms that can be used for sources.