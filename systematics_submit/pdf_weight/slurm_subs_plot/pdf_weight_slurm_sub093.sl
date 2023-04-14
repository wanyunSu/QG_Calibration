#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 00:10:00
#SBATCH -J qgtagging_plot_pdf_weight093
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --account=atlas
#SBATCH --mem=6G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/slurm_logs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/make_plot_pdf_weight.sh 93
