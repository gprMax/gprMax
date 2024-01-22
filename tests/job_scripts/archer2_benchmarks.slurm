#!/bin/bash
#SBATCH --job-name=gprMax-benchmarks
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --qos=serial
#SBATCH --output=output/archer2/rfm_bench_%J.out

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

source ../.venv/bin/activate

reframe -C configuration/archer2_settings.py -c reframe_benchmarks.py -c base_tests.py -r --performance-report

sacct --format=JobID,State,Submit,Start,End,Elapsed,NodeList --units=M -j $SLURM_JOBID
