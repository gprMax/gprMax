*************************************
Machine Learning based Forward Solver
*************************************

This folder contains the essential files for using a near-real time Machine Learning (ML) based Forward Solver. The ML framework uses an innovative training method that combines a data compression technique (such as PCA or SVD) and a large data set of modeled GPR responses from gprMax. The ML-based forward solver is parameterized for a specific GPR application, but the framework can be applied to many different classes of GPR problems. 

* The sample ML notebook (`ML.ipynb <https://github.com/utsav-akhaury/gprMax/blob/devel/ML/ML.ipynb>`_) serves as a template for using the newly added ``Random Parameter Generation Feature`` in conjunction with the chosen ML scheme. Depending on the scenario being modelled, the user may need to change the hyper-parameters (or use a different ML model) to better fit the data
* `ML_utilities.py <https://github.com/utsav-akhaury/gprMax/blob/devel/ML/ML_utilities.py>`_ contains a few helpful functions required for using the ML feature
* ``sample_models`` contains the gprMax input file for the specific model that was used for near-real time prediction in the sample ML solver notebook

Results
-------

A summary of the performance of different ML schemes on our sample test dataset (after PCA compression):

(``NMSE = Normalized Mean Squared Error`` - Lower NMSE implies better performance)

============================================== =========================== ==================
Method                                         NMSE (on 1250 test samples) Training Time (s)
============================================== =========================== ==================
Random Forest                                  0.0182                      2.8
Random Forest + Chain Regression               0.0809                      179.5
XGBoost                                        0.0285                      8.8
XGBoost + Chain Regression                     0.1011                      17.4
SVM                                            0.2488                      25.7
SVM + Chain Regression                         0.2487                      22.3
SGDRegressor                                   0.2394                      0.1
SGDRegressor  + Chain Regression               0.2402                      0.4
Gradient Boosting Regressor                    0.1191                      17.4
Gradient Boosting Regressor + Chain Regression 0.1302                      74.1
============================================== =========================== ==================
