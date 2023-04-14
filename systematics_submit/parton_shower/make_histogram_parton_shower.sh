#!/usr/bin/env bash
# This type of systematics only contain small number of variants 
# One can run it interactively

syst_identifier=parton_shower
systs_subtypes="herwigangle herwigdipole"

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml

for systs_subtype in $systs_subtypes
do
    python ${workdir}/make_histogram_new.py --write-log \
    --input-mc-path /global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/$systs_subtype \
    --do-systs --systs-type ${syst_identifier} --systs-subtype $systs_subtype \
    --output-path ${output} \
    --do-plotting 
done