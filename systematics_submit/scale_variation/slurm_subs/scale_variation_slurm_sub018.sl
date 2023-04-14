#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 1:00:00
#SBATCH -J qgtagging_scale_variation018
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=80G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/slurm_logs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/make_histogram_scale_variation.sh 18
