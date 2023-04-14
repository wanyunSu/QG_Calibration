workdir=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/
merged_hist_path=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal/

for reweighting in none quark gluon 
do
    echo 'doing plot for ${reweighting}...'
    python -u ${workdir}/core/Calculate_SF.py \
    --path-mc ${merged_hist_path}/MC_merged_hist.pkl \
    --path-data ${merged_hist_path}/Data_merged_hist.pkl \
    --period ADE --reweighting  ${reweighting}\
    --output-path ${workdir}/results_CalculateSF
done