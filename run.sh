#!/bin/bash 

#PBS -l ncpus=48
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -l jobfs=128GB
#PBS -q gpuvolta
#PBS -P ry05
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/ry05+scratch/ry05
#PBS -l wd

module load python3/3.9.2 pytorch/1.9.0
python3 main.py $PBS_NCPUS > /g/data/ry05/$USER/job_logs/$PBS_JOBID.log