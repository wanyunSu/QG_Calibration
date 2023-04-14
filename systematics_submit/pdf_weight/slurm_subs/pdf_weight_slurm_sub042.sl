#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 1:00:00
#SBATCH -J qgtagging_pdf_weight042
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=100G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/slurm_logs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/systematics_submit/pdf_weight/make_histogram_pdf_weight.sh 42
