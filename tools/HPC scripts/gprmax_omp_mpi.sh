#!/bin/sh
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
#$ -pe mpi 800

### Job script name:
#$ -N gprmax_omp_mpi.sh
#####################################################################################

### Initialise environment module
. /etc/profile.d/modules.sh

### Load Anaconda environment for gprMax, i.e. Python 3 and required packages
module load anaconda
source activate gprMax

### Load OpenMPI
module load openmpi

### Set number of OpenMP threads
export OMP_NUM_THREADS=8

### Run gprMax with input file
cd $HOME/gprMax
mpiexec -n 100 python -m gprMax mymodel.in -n 100 -mpi