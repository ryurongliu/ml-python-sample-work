#!/bin/sh

#SBATCH --account=astro
#SBATCH --job-name=5000e_1000y_0vinc    ####CHANGE THIS!
#SBATCH -c 1
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --output=%x_%a.out
#SBATCH --array=1-20
module load anaconda

python 5000e_1000y_0vinc.py $SLURM_ARRAY_TASK_ID   #####AND CHANGE PY FILE NAME!!
