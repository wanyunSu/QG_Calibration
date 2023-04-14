#!/usr/bin/env bash
systs_subtype=pdf_weight

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml


python ${workdir}/make_histogram_new.py --write-log \
--do-systs --systs-type ${systs_subtype} --systs-subtype $1 \
--output-path ${output} \
--do-plotting 