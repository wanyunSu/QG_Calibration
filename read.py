import joblib
with open('/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/syst_uncertainties/syst_total.pkl','rb') as f:
    obj=joblib.load(f)

print(obj)
