Toolboxes is a sub-package where useful Python modules contributed by users are stored.

*****************
gprMax on Jupyter
*****************

Information
===========

This package is intended to help users run gprMax on a Jupyter Notebook environment.

Package contents
================

install_notebook_dependencies.py
--------------------------------

This module installs the gprMax-devel kernel in Jupyter notebook. After installing the gprMax through the normal installation, ensuring that Miniconda and GCC (for Linux/Mac) or Microsoft Build Tools for Visual Studio 2022 (for Windows) are installed, we run this module to ensure that the dependencies for running models on Jupyter Notebook are installed. 

.. code-block:: none

    python toolboxes.gprMaxOnJupyter.install_notebook_dependencies.py

example_notebooks
-----------------

This sub-package contains examples of several models run on Jupyter Notebooks with the gprMax-devel kernel. The notebooks of the models are: 

* ``cylinder_Ascan_2D`` contains the following files :
	* ``cylinder_Ascan_2D.in`` : This is the input file containing instructions on how to build the model.
	* ``cylinder_Ascan_2D.h5`` : This is the output file after running the model on the Jupyter Notebook.
	* ``cylinder_half_space.vti`` : This is the 3D geometry output file of the A_Scan which is used to visualize the model in Jupyter Notebook.
	* ``cylinder_Ascan_2D_notebook.ipynb`` : This is the Jupyter Notebook on which the model of the `A_Scan of a cylinder` is built and run. The A_Scan is plot in this notebook along with the 3D model.

* ``cylinder_Bscan_2D`` contains the following files :
	* ``cylinder_Bscan_2D.in`` : This is the input file containing instructions on how to build the model.
	* ``cylinder_Bscan_2D*.h5`` : These are the output files after running the model on the Jupyter Notebook. These range from 1 to 10. 
	* ``cylinder_Bscan_2D_merged.h5`` : This is the merged output file from the 10 output files after running the model.
	* ``cylinder_Bscan_2D5.vti`` : This is the 3D geometry output file of the B_Scan which is used to visualize the model in Jupyter Notebook.
	* ``cylinder_Bscan_2D_notebook.ipynb`` : This is the Jupyter Notebook on which the model of the `B_Scan of a cylinder` is built and run. The B_Scan is plot in this notebook along with the 3D model.

* ``GSSI_1500_antenna_model`` contains the following files :
	* ``GSSI_1500_antenna_Bscan.in`` : This is the input file containing instructions on how to build the model.
	* ``GSSI_1500_antenna_Bscan.h5`` : This is the output file after running the model on the Jupyter Notebook.
	* ``antenna_like_GSSI_1500.vti`` : This is the 3D geometry output file of the GSSI 1500 antenna which is used to visualize the model in Jupyter Notebook.
	* ``GSSI_1500_antenna_Bscan.ipynb`` : This is the Jupyter Notebook on which the model of the `GSSI 1500 antenna` is built and run. The B_Scan is plot in this notebook along with the 3D model.

* ``hertzian_dipole_hs`` contains the following files :
	* ``hertzian_dipole_hs.in`` : This is the input file containing instructions on how to build the model.
	* ``hertzian_dipole_hs.h5`` : This is the output file after running the model on the Jupyter Notebook.
	* ``hertzian_dipole_hs_notebook.ipynb`` : This is the Jupyter Notebook on which the model of a `hertzian dipole in free space` is built and run. The A_Scan of this is plot in this notebook.

* ``magnetic_dipole_fs`` contains the following files :
	* ``magnetic_dipole_hs.in`` : This is the input file containing instructions on how to build the model.
	* ``magnetic_dipole_hs.h5`` : This is the output file after running the model on the Jupyter Notebook.
	* ``magnetic_dipole_hs_notebook.ipynb`` : This is the Jupyter Notebook on which the model of a `magnetic dipole in free space` is built and run. The A_Scan of this is plot in this notebook.