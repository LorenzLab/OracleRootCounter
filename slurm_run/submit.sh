#!/bin/bash

#SBATCH -A ACD112211
#SBATCH -p ct56
#SBATCH --ntasks=20
#SBATCH --nodes=20
#SBATCH -c 56


srun mp_dataset_gen.py
