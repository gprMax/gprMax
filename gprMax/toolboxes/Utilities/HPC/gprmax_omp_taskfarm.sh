#!/bin/bash
#####################################################################################
### Change to current working directory:
#$ -cwd

### Specify runtime (hh:mm:ss):
#$ -l h_rt=01:00:00

### Email options:
#$ -m ea -M joe.bloggs@email.com

### Resource reservation:
#$ -R y

### Parallel environment ($NSLOTS):
#$ -pe mpi 176

### Job script name:
#$ -N gprmax_omp_taskfarm.sh
#####################################################################################

### Initialise environment module
set -e
. /etc/profile.d/modules.sh

### Load and activate Anaconda environment for gprMax, i.e. Python 3 and required packages
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gprMax-v4

### Load OpenMPI
module load openmpi

### Set number of OpenMP threads per MPI task (each gprMax model)
export OMP_NUM_THREADS=16

### Submit from the working directory containing mymodel.in (#$ -cwd above).
mpirun -n 11 python -m gprMax mymodel.in -n 10 --taskfarm
