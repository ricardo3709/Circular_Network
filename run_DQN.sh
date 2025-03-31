#!/bin/bash 

#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=128GB
#PBS -q gpuvolta
#PBS -P ry05
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/ry05+scratch/ry05
#PBS -l wd

module load python3/3.9.2 pytorch/1.9.0
source venv/RL_3.9.2/bin/activate
python3 main.py $PBS_NCPUS > ./job_${PBS_JOBID}.log 2>&1