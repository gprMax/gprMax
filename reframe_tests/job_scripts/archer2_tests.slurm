#!/bin/bash
#SBATCH --job-name=gprMax-tests
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --output=output/archer2/rfm_tests_%J.out

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

module load cray-python

source ../.venv/bin/activate

reframe -C configuration/archer2_settings.py -c tests/ -r "$@"

sacct --format=JobID,State,Submit,Start,End,Elapsed,NodeList --units=M -j $SLURM_JOBID
