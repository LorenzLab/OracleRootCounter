#!/bin/bash

#SBATCH -A ACD112211
#SBATCH --ntasks=4

srun mp_dataset_gen.py
