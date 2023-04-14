#!/usr/bin/env bash

workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/test_makehistogram/

source /global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate ml
python ${workdir}/make_histogram.py --write-log \
--do-systs --systs-type JESJER --systs-subtype $1 \
--output-path ${output}/$1 

# python ${workdir}/make_histogram.py --write-log \
# --output-path ${output}

python ${workdir}/final_plotting.py --write-log \
--output-path ${output}