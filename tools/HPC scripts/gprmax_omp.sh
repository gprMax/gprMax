#!/bin/sh
#####################################################################################
### Change to current working directory:
#$ -cwd

### Specify runtime (hh:mm:ss):
#$ -l h_rt=01:00:00

### Email options:
#$ -m ea -M joe.bloggs@email.com

### Parallel environment ($NSLOTS):
#$ -pe sharedmem 16

### Job script name:
#$ -N gprmax_omp.sh
#####################################################################################

### Initialise environment module
. /etc/profile.d/modules.sh

### Load and activate Anaconda environment for gprMax, i.e. Python 3 and required packages
module load anaconda
source activate gprMax

### Set number of OpenMP threads for each gprMax model
export OMP_NUM_THREADS=16

### Run gprMax with input file
cd $HOME/gprMax
python -m gprMax mymodel.in -n 10