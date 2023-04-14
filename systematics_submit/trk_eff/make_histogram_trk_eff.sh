#!/usr/bin/env bash
# This type of systematics only contain small number of variants 
# One can run it interactively

systs_type=trk_eff
systs_subtypes="jet_nTracks_sys_eff jet_nTracks_sys_fake"

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml

for systs_subtype in $systs_subtypes
do
    python ${workdir}/make_histogram_new.py --write-log \
    --do-systs --systs-type ${systs_type} --systs-subtype $systs_subtype \
    --output-path ${output} \
    --do-plotting 
done