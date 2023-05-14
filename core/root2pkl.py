import argparse
import os, sys
import logging
import joblib 
import numpy as np
import pandas as pd 
import uproot 
import awkward as ak
from pathlib import Path
import re
from .utils import check_inputpath, check_outputpath, logging_setup
from .utils import trk_eff_uncertainties, JESJER_uncertainties, pdf_weight_uncertainties, scale_variation_uncertainties

luminosity_periods = {
    "A" : 36000,#3244.54 + 33402.2
    "D" : 44630,#44630.6
    "E" : 58791 #58791.6 #https://twiki.cern.ch/twiki/bin/viewauth/Atlas/LuminosityForPhysics
}

MC_identifier_config = {
    "pythia" :{
        "DSID_JZ" : dict(zip(np.arange(364700, 364710, 1, dtype = int), np.arange(0, 10, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(0, 10, 1,dtype = int), 
                          np.array([7.8050E+07, 7.8050E+07, 2.4330E+06, 2.6450E+04, 2.5461E+02, 4.5532E+00, 2.5754E-01, 1.6215E-02, 6.2506E-04, 1.9639E-05])*1E3)), # pb
        "eff" : dict(zip(np.arange(0, 10, 1, dtype = int), 
                         np.array([9.753257E-01, 2.442497E-02, 9.863129E-03, 1.165838E-02, 1.336560E-02, 1.452648E-02, 9.471878E-03, 1.1097E-02, 1.015436E-02, 1.2056E-02])))
    },
    "herwigangle" :{
        "DSID_JZ" : dict(zip(np.arange(364922, 364930, 1, dtype = int), np.arange(2, 10, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(2, 10, 1,dtype = int), 
                          np.array([3.4657E+06, 3.7009E+04, 3.5364E+02, 6.0338E+00, 3.2248E-01, 1.915E-02, 7.0007E-04, 2.1245E-05])*1E3)), # pb
        "eff" : dict(zip(np.arange(2, 10, 1, dtype = int), 
                         np.array([5.2691E-03, 6.0371E-03, 7.3993E-03, 8.4098E-03, 5.7036E-03, 7.0481E-03, 6.7365E-03, 8.1683E-03])))
    },
    "herwigdipole" :{
        "DSID_JZ" : dict(zip(np.arange(364902, 364910, 1, dtype = int), np.arange(2, 10, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(2, 10, 1,dtype = int), 
                          np.array([3.3961E+06, 3.6974E+04, 3.5341E+02, 6.0337E+00, 3.2249E-01, 1.9160E-02, 7.0035E-04, 2.1235E-05])*1E3)), # pb
        "eff" : dict(zip(np.arange(2, 10, 1, dtype= int), 
                         np.array([4.4714E-03, 5.5708E-03, 6.9756E-03, 7.9811E-03, 5.4878E-03, 6.8288E-03, 6.6413E-03, 8.2273E-03])))
    },
    "sherpa" :{
        "DSID_JZ" : dict(zip(np.arange(364677, 364686, 1, dtype = int), np.arange(1, 10, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(1, 10, 1,dtype = int), 
                          np.array([2.1406E+07, 8.9315E+06, 9.2751E+04, 5.5101E+02, 1.6315E+00, 1.2842E-01, 2.7212E-02, 2.0582E-04, 3.5684E-05])*1E3)), # pb
        "eff" : dict(zip(np.arange(1, 10, 1, dtype = int), 
                         np.array([1.441444E-03, 5.117554E-03, 5.651610E-04, 1.497205E-03, 2.4211E-02, 1.0823E-02, 3.6186E-03, 1.5963E-02, 3.3036E-03])))
    },
    "sherpalund" :{
        "DSID_JZ" : dict(zip(np.arange(364686, 364695, 1, dtype = int), np.arange(1, 10, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(1, 10, 1,dtype = int), 
                          np.array([2.1406E+07, 8.9314E+06, 9.2750E+04, 5.5101E+02, 1.6312E+00, 1.2841E-01, 2.7211E-02, 2.0584E-04, 3.5683E-05])*1E3)), # pb
        "eff" : dict(zip(np.arange(1, 10, 1, dtype = int), 
                         np.array([1.4129E-03, 5.1181E-03, 5.6410E-04, 1.4985E-03, 2.4259E-02, 1.0844E-02, 3.6264E-03, 1.6003E-02, 3.3126E-03])))
    },
    "powhegpythia" :{
        "DSID_JZ" : dict(zip(np.arange(600624, 600632, 1, dtype = int), np.arange(1, 9, 1, dtype = int))),
        "xsec" : dict(zip(np.arange(1, 9, 1,dtype = int), 
                          np.array([5.4106E+08, 5.5822E+07, 1.8689E+06, 1.7143E+05, 5.1827E+03, 2.7737E+02, 2.6527E+01, 7.6882E-02])*1E3)), # pb
        "eff" : dict(zip(np.arange(1, 9, 1, dtype= int), 
                         np.array([4.5184E-03, 3.6883E-04, 1.3383E-04, 1.6146E-05, 1.0020E-05, 6.7286E-06, 5.1581E-06, 6.4484E-05])))
    }
}


label_pt_bin = np.array([500, 600, 800, 1000, 1200, 1500, 2000])

def read_SumofWeights_Period(sample_folder_path:Path, MC_identifier:str, period:str):
    """Sum of the first value of unmerged hist.root files and return a numpy array. The length depends on the JZ slices. 

    Args:
        sample_folder_path (Path()): The Path to the rucio downloaded folders.
        period (String): Choose from ["A", "D", "E"], corresponding to periods. 

    Returns:
        Numpy Array: This array is the some of weights from different JZ slices. 
    """
    if not period in ["A", "D", "E"]:
        raise Exception(f'Period {period} not in supported periods. Currently supported: ["A", "D", "E"]')
    
    file_path = sample_folder_path / f'SumofWeights_{period}.pkl'

    if not file_path.exists():
        period_JZslice = sorted(sample_folder_path.rglob(f"*JZ*_hist"))
        # if len(period_JZslice) == 0:
            # period_JZslice = sorted(sample_folder_path.rglob(f"user.*{str.lower(period)}.mc16_13TeV.36470?.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ?WithSW_hist"))
        if len(period_JZslice) == 0:
            raise Exception(f"No hist files found in {sample_folder_path} ")

        period_JZ_sum = {}
        logging.debug("read_SumofWeights_Period: Start reading sum of weights")
        for i, dir in enumerate(period_JZslice):
            # do this for each JZ slice 
            logging.debug(f"\t {dir} \n")
            match = re.search(r"JZ(\d+)", dir.name)
            if match:
                # get the number after "JZ"
                number = match.group(1)
                number = int(number)
                sum_JZ_slice = 0 
                for file in sorted(dir.glob("*.hist-output.root")):
                    # e.g. for JZ1, loop over all the hist files  
                    sum_JZ_slice += uproot.open(file)['histoEventCount'].values()[0]
            
                period_JZ_sum[number] = sum_JZ_slice
            else: 
                raise Exception("Not found the JZ slice number based on the folder name!")
            
        joblib.dump(period_JZ_sum, file_path)
    else:
        period_JZ_sum = joblib.load(file_path)
    
    return period_JZ_sum

def apply_cut(sample,is_MC=False):
    # event trigger selection 
    event_trigger_idx = sample["jet_fire"] == 1
    sample = sample[event_trigger_idx]

    # pT cut 
    pt_cut_idx = sample["jet_pt"][:,1] > 500000
    sample = sample[pt_cut_idx]

    pt_max_cut_idx = sample["jet_pt"][:,0] < 2000000 
    sample = sample[pt_max_cut_idx]

    # ratio < 1.5
    sample = sample[sample["jet_pt"][:,0]/sample["jet_pt"][:,1] < 1.5]

    # eta cut 
    sample = sample[np.abs(sample["jet_eta"][:,0]) < 2.1]
    sample = sample[np.abs(sample["jet_eta"][:,1]) < 2.1]

    sample = sample[np.abs(sample["event_weight"]) < 100]
    sample = sample[sample["jet_cleaning"][:,0] == 2]
    sample = sample[sample["jet_cleaning"][:,1] == 2]
    sample = sample[sample["jet_nTracks"][:,0] > 1]
    sample = sample[sample["jet_nTracks"][:,1] > 1]
    if is_MC:
        sample = sample[sample["jet_PartonTruthLabelID"][:,0] > 0]
        sample = sample[sample["jet_PartonTruthLabelID"][:,1] > 0]

    return sample 

def root2pkl(root_file_path, is_MC=True, output_path=None, 
             do_systs=False, systs_type=None, systs_subtype=None,
             MC_identifier = 'pythia', if_save = False):
    """Give me a root file, flatten it into pandas format.

    Args:
        root_file_path (str or Path): the file path to a root file.
        output_path (str or Path, optional): the output path for a pkl file. Defaults to None.
        verbosity (int, optional): log level. Defaults to 2.
        write_log (bool, optional): if we write the log to a file. Defaults to False.

    Raises:
        Exception: root file not found. 
        Exception: root file cannot open.

    Returns:
        DataFrame: pandas DataFrame with flatten events.
    """
    root_file_path = check_inputpath(root_file_path)

    if output_path is None:
        output_path = root_file_path.parent
    else:
        output_path = check_outputpath(output_path)
    # logging_setup(verbosity, write_log, output_path) # This is done in make_histogram_new.py 

    try:
        logging.debug(f"Processing file: {root_file_path.stem}")
        sample = uproot.open(root_file_path)
    except Exception as Argument:
        raise Exception(f"Open root file failed! {root_file_path}")

    ttree_name = 'nominal'
    ## if doing JESJER, overwrite tree name 
    if is_MC and systs_type == 'JESJER':
        if not systs_subtype in JESJER_uncertainties:
            raise Exception(f"{systs_subtype} is not avaiale in valid JESJER types.")
        ttree_name = systs_subtype

    #branch_names = ["run", "event", "pu_weight", "jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]
    branch_names = ["run", "event", "pu_weight", "jet_fire", "jet_pt", "jet_eta", "jet_nTracks", "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID","jet_cleaning"]
    if is_MC: # Only do the systematics for MC
        if do_systs and systs_type == 'trk_eff':
            if not systs_subtype in trk_eff_uncertainties:
                raise Exception(f"{systs_subtype} is not avaiale in valid track efficiency types.")
            branch_names = ["run", "event", "pu_weight", "jet_fire", "jet_pt", "jet_eta", systs_subtype, "jet_trackWidth", "jet_trackC1", "jet_trackBDT", "jet_PartonTruthLabelID"]

        if do_systs and systs_type == 'pdf_weight':
            if not systs_subtype in pdf_weight_uncertainties:
                raise Exception(f"{systs_subtype} is not avaiale in valid pdf weight types.")
            branch_names.append('pdf_weight')

        if do_systs and systs_type == 'scale_variation':
            if not systs_subtype in scale_variation_uncertainties:
                raise Exception(f"{systs_subtype} is not avaiale in valid pdf weight types.")
            branch_names.append('mcEventWeightsVector')
            branch_names.append('mconly_weight')
    try:
        sample_ak = sample[ttree_name].arrays(branch_names, library='ak')
    except uproot.exceptions.KeyInFileError:
        logging.warning(f"{root_file_path} no TTree {ttree_name}!")
        return 

    if len(sample_ak) == 0:
        logging.warning(f"{root_file_path} is empty")
        return  # Notice this will put a None as return 

    is_Data = np.all(sample_ak['jet_PartonTruthLabelID'][0]==-9999) # Check the first event in sample and assert if it's data
    assert is_MC == (not is_Data)
    period_folder = root_file_path.parent.parent
    if is_MC:
        period_search_pattern = f"{MC_identifier}[A,D,E]"
        match = re.search(period_search_pattern, period_folder.stem)
        if match:
            period = match.group()[-1]
            assert period in ["A", "D", "E"]
        else:
            print(f"period_folder = {period_folder.stem}, period_search_pattern={period_search_pattern}")
            raise Exception("Period not found by the match pattern. ")
        
        sum_of_weights = read_SumofWeights_Period((period_folder.parent/ f'{MC_identifier}{period}_hist'), MC_identifier, period)

        DSID_JZ = MC_identifier_config[MC_identifier]["DSID_JZ"]
        xsec = MC_identifier_config[MC_identifier]["xsec"]
        eff = MC_identifier_config[MC_identifier]["eff"]

        JZ_slice_number = DSID_JZ[sample_ak.run[0]] # JZ slice for each event
        event_weight = luminosity_periods[period] * sample_ak["pu_weight"] * xsec[JZ_slice_number] * eff[JZ_slice_number]*0.78/ sum_of_weights[JZ_slice_number] # Changed to key, lookup the var by JZ_slice_number 
        # pu_weight is already multiplied by mcEventWeight in MonoJetx.cxx 
        if do_systs and systs_type == 'pdf_weight':
            event_weight = event_weight * sample_ak["pdf_weight"][:, int(systs_subtype)]
        
        if do_systs and systs_type == 'scale_variation':
            event_weight = event_weight * sample_ak["mcEventWeightsVector"][:, int(systs_subtype)] / sample_ak['mconly_weight'][:]

    else:
        period_search_pattern = r'data(.+)$'
        period = re.search(period_search_pattern, period_folder.stem).group((1))
        assert period in ["1516", "17", "18"]
        event_weight = ak.ones_like(sample_ak['event'], dtype=np.float32) # For data, event weight is always 1 

        #if period == "18":
        #   event_weight = event_weight * 59.45/39.91
           

    sample = ak.with_field(base = sample_ak, what = event_weight, where = "event_weight")

    sample = apply_cut(sample)

    if len(sample) == 0:
        logging.warning(f"{root_file_path} is empty after cut")
        return
    
    sample_pd = ak.to_pandas(sample)
    sample_dijet_pd = sample_pd.loc[(slice(None), slice(0,1)), :]
    sample_dijet_pd = sample_dijet_pd.drop(['pu_weight', 'jet_fire'], axis = 1)

    pt_idx = sample_dijet_pd.columns.get_loc('jet_pt')
    eta_idx = sample_dijet_pd.columns.get_loc('jet_eta')

    # Rename the read systs_subtype to jet_nTracks for following manipulation 
    if do_systs and systs_type == 'trk_eff':
        sample_dijet_pd.rename(columns={systs_subtype:'jet_nTracks'}, inplace=True)
        
    # sample_dijet_pd.iloc[:, pt_idx] = sample_dijet_pd.iloc[:, pt_idx] / 1000
    sample_dijet_pd['jet_pt'] = sample_dijet_pd['jet_pt'].div(1000)
    sample_dijet_np = sample_dijet_pd.to_numpy().reshape((len(sample_dijet_pd)//2, 2, len(sample_dijet_pd.columns)))
    # assert np.allclose(sample_pd.loc[0]['jet_eta'].to_numpy(), sample_dijet_np[0])

    #### Add two labels, is_leading and is_forward 
    forward_idx = np.argmax(np.abs(sample_dijet_np[:,:,eta_idx]), axis=1) # compare abs eta of jets inside events
    is_forward = np.zeros((len(sample_dijet_np),2))
    is_forward[np.arange(len(is_forward)), forward_idx] = 1
    is_leading = np.zeros((len(sample_dijet_np),2))
    is_leading[:, 0] = 1
    sample_dijet_np_label = np.concatenate((sample_dijet_np, np.broadcast_to(is_forward[:,:,None], (sample_dijet_np.shape[:2] + (1,)))), axis = 2)
    sample_dijet_np_label = np.concatenate((sample_dijet_np_label, np.broadcast_to(is_leading[:,:,None], (sample_dijet_np_label.shape[:2] + (1,)))), axis = 2)
    
    sample_pd_label = pd.DataFrame(sample_dijet_np_label.reshape(-1, sample_dijet_np_label.shape[-1]), columns = sample_dijet_pd.columns.to_list() + ["is_forward", "is_leading"], dtype=np.float64)

    #### Add pt label, pt_idx
    sample_pd_label['pt_idx'] = pd.cut(x=sample_pd_label['jet_pt'], bins=label_pt_bin, right=False, labels=False)

    if is_MC:
        #### Add parton truth label, for ML training purpose 
        #### Do this only for MC 
        sample_pd_label['target'] = '-'
        target_idx = sample_pd_label.columns.get_loc('target')
        gluon_idx = sample_pd_label['jet_PartonTruthLabelID'] == 21
        quark_idx = ((sample_pd_label['jet_PartonTruthLabelID'] > 0) & (sample_pd_label['jet_PartonTruthLabelID'] < 10))

        sample_pd_label.iloc[gluon_idx, target_idx] = 1
        sample_pd_label.iloc[quark_idx, target_idx] = 0

    if if_save:
        joblib.dump(sample_pd_label, output_path / (root_file_path.stem + ".pkl"))
    return sample_pd_label



if __name__ == '__main__':
    # For test case.
    logging_setup(verbosity=4, if_write_log=False, output_path=None)
    # MC_test_file_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/powhegpythia/powhegpythiaA/user.wasu.JanP20A.mc16_13TeV.600628.PhPy8EG_jj_JZ5_minitrees.root/user.wasu.32118723._000001.minitrees.root'
    # root2pkl(MC_test_file_path, is_MC=True, MC_identifier="powhegpythia")

    MC_test_file_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/pythiaA/user.wasu.Dec11a.mc16_13TeV.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_minitrees.root/user.wasu.31564717._000001.minitrees.root'
    root2pkl(MC_test_file_path, is_MC=True, MC_identifier="pythia", output_path='/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/test_root2pkl', if_save=True)
    
    test_file_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New/data/data18/user.wasu.Oct18.data18_13TeV.periodB.physics_Main_minitrees.root/user.wasu.30894553._000005.minitrees.root'
    root2pkl(test_file_path, is_MC=False)
    pass
