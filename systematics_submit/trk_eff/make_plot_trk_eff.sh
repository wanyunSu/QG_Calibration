#!/usr/bin/env bash
# This type of systematics only contain small number of variants 
# One can run it interactively

systs_type=trk_eff
systs_subtypes="jet_nTracks_sys_fake jet_nTracks_sys_fake"

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml

for systs_subtype in $systs_subtypes
do
    echo ${systs_subtype}
    python ${workdir}/make_plot_new.py --input-path ${output}/${systs_type}/${systs_subtypes} \
    --do-systs --systs-type ${systs_type} --systs-subtype ${systs_subtype} --write-log 
done