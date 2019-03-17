conda update conda --yes
conda install git  --yes

conda env create -f conda_env.yml

source activate gprMax
python setup.py build
python setup.py install
conda init
