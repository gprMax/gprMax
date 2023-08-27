#!/bin/bash
conda init 
conda update conda --yes
conda install git --yes
conda env create -f conda_env.yml
conda activate gprMax-devel
python setup.py build
python setup.py install
