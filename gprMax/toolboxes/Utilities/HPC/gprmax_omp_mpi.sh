#!/bin/bash

### Job script name:
#SBATCH --job-name="gprMax MPI demo"

### Number of MPI tasks:
#SBATCH --ntasks=8

### Number of CPUs (OpenMP threads) per task:
#SBATCH --cpus-per-task=16

### Runtime limit:
#SBATCH --time=0:10:0

### Partition and quality of service to use (these control the type and
### amount of resources allowed to request):
#SBATCH --partition=standard
#SBATCH --qos=standard

### Hints to control MPI task layout:
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block


# Set number of OpenMP threads from SLURM environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Load system modules
module load PrgEnv-gnu
module load cray-python

# Load Python virtual environment
source .venv/bin/activate

# Run gprMax with input file
srun python -m gprMax my_model.in --mpi 2 2 2
