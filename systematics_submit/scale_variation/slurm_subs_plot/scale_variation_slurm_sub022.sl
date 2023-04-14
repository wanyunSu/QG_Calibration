#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 00:10:00
#SBATCH -J qgtagging_plot_scale_variation022
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --account=atlas
#SBATCH --mem=6G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/slurm_logs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/scale_variation/make_plot_scale_variation.sh 22
