*************************************
Machine Learning based Forward Solver
*************************************

This folder contains the essential files for using a near-real time Machine Learning (ML) based Forward Solver. The ML framework uses an innovative training method that combines a data compression technique (such as PCA or SVD) and a large data set of modeled GPR responses from gprMax. The ML-based forward solver is parameterized for a specific GPR application, but the framework can be applied to many different classes of GPR problems. 

* The sample ML notebook (`ML.ipynb <https://github.com/utsav-akhaury/gprMax/blob/devel/ML/ML.ipynb>`_) inside this folder serves as a template for using the newly added ``Random Parameter Generation Feature`` in conjuction with the chosen ML scheme. Depending on the scenario being modelled, the user may need to change the hyper-parameters (or use a different ML model) to better fit the data
* `ML_utils <https://github.com/utsav-akhaury/gprMax/blob/devel/ML/ML_utilities.py>`_ contains a few helpful functions required for using the ML feature
* ``sample_models`` contains the gprMax input file for the specific model that was used for near-real time prediction in the sample ML solver notebook