#!/bin/bash
#####################################################################################
### Change to current working directory:
#$ -cwd

### Specify runtime (hh:mm:ss):
#$ -l h_rt=01:00:00

### Parallel environment ($NSLOTS):
#$ -pe sharedmem 16

### Job array and task IDs
#$ -t 1-10

### Job script name:
#$ -N gprmax_omp_jobarray.sh
#####################################################################################

### Initialise environment module
set -e
. /etc/profile.d/modules.sh

### Load and activate Anaconda environment for gprMax, i.e. Python 3 and required packages
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gprMax-v4

### Set number of OpenMP threads for each gprMax model
export OMP_NUM_THREADS=16

### Submit from the working directory containing mymodel.in (#$ -cwd above).
### One model per scheduler task; -n is a count, not the array's total size.
### A one-model run needs an explicit unique output prefix for each task.
python -m gprMax mymodel.in -n 1 -i "${SGE_TASK_ID:?Grid Engine must supply SGE_TASK_ID}" -o "mymodel_${SGE_TASK_ID}"
