#!/bin/bash

#SBATCH -C haswell
#SBATCH -t 02:00:00
#SBATCH -J qgtagging
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=50G
#SBATCH --output=./test_slurm3/slurm-%j.out
#SBATCH --error=./test_slurm3/slurm-%j.err

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
conda activate ml 
srun --cpu_bind=none python -u make_histogram_new.py --output-path /global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/test_slurm3
