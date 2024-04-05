#!/bin/bash

#SBATCH -A ACD112211
#SBATCH -p ctest
#SBATCH -n 20
#SBATCH -N 20
#SBATCH -c 56


srun mp_dataset_gen.py
