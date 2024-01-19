#!/bin/bash
#SBATCH --job-name=gprMax-map
#SBATCH --time=00:20:0
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=output/archer2/map_%J.out

# USAGE: sbatch -c [ompthreads] job_scripts/archer2_map_single_node.slurm [domain]

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load arm/forge

source ../.venv/bin/activate

mkdir -p profile/archer2/

map -o="profile/archer2/gprMax_${1}d_${SLURM_CPUS_PER_TASK}t_$(date +%F_%H-%M)" --mpi=slurm --mpiargs="--hint=nomultithread --distribution=block:block" --profile python -m gprMax --log-level 25 src/benchmark_model_$1.in

sacct --format=JobID,State,Submit,Start,End,Elapsed,NodeList --units=M -j $SLURM_JOBID