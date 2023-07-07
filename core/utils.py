import numpy as np
import pandas as pd
from pathlib import Path
import hist
from hist import Hist
from uncertainties import ufloat, unumpy
import joblib
import re


###########################################################
###################Some global variables###################
###########################################################

label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]
# label_var = ["pt", "eta", "ntrk", "width", "c1", "bdt", "newBDT"]
label_var = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1','GBDT_newScore']
label_leadingtype = ["LeadingJet", "SubLeadingJet"]
label_etaregion = ["Forward", "Central"]
label_jettype = ["Gluon", "Quark", "B_Quark", "C_Quark"]
label_jettype_data = ["Data"]

reweighting_vars = ['jet_nTracks', 'GBDT_newScore'] 
nominal_keys = [reweighting_var + '_quark_reweighting_weights' for reweighting_var in reweighting_vars]

label_var_map = {
    'pt':'jet_pt',
    'eta':'jet_eta',
    'ntrk':'jet_nTracks', 
    'width':'jet_trackWidth',
    'c1':'jet_trackC1', 
    'bdt':'jet_trackBDT', 
    'newBDT':'GBDT_newScore'
}

is_leading_map = {
    "LeadingJet": [1],
    "SubLeadingJet": [0],
}

is_forward_map = {
    "Forward": [1],
    "Central": [0],
}

label_jettype_map = {
    "Gluon" : [21], 
    "Quark" : [1, 2, 3],
    "B_Quark" : [5],
    "C_Quark" : [4],
    "Data" : [-9999.0]
}

HistBins = {
    'jet_pt' : np.linspace(500, 2000, 61),
    'jet_eta' : np.linspace(-2.5, 2.5, 51), 
    'jet_nTracks' : np.linspace(0, 60, 61),
    'jet_trackWidth' : np.linspace(0, 0.4, 61),
    'jet_trackC1' : np.linspace(0, 0.4, 61),
    'jet_trackBDT' : np.linspace(-1.0, 1.0, 101),
    #'GBDT_newScore' : np.linspace(-2.0, 2.0, 51),
    'GBDT_newScore' : np.linspace(-5.0, 5.0, 101),
    #'GBDT_newScore' : np.linspace(0, 1.0, 51),
}


def logging_setup(verbosity, if_write_log, output_path, filename="make_histogram"):
    import logging
    log_levels = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARN,
        3: logging.INFO,
        4: logging.DEBUG,
    }
    if if_write_log:
        check_outputpath(output_path)
        logging.basicConfig(filename=output_path / f'log.{filename}.txt', filemode='w', level=log_levels[verbosity], 
                            format='%(asctime)s  %(levelname)s  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(level=log_levels[verbosity], 
                            format='%(asctime)s  %(levelname)s  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


###########################################################
###################Some useful functions###################
###########################################################
def check_inputpath(input_path):
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    if not input_path.exists():
        raise Exception(f"File {input_path} not found. ")
    return input_path

def check_outputpath(output_path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path 

def normalize_hist(_hist):
    if _hist.sum().value != 0:
        area = np.sum(_hist.values()) * _hist.axes[0].widths
        _normed = _hist / area
    else:
        _normed = _hist
    return _normed
    

def make_hist(values, bins, weights):
    # assuming bins numpy array with (start, stop, n_edges)
    _hist = Hist(hist.axis.Regular(bins=len(bins)-1, start=bins[0], stop=bins[-1], overflow=True, underflow=True), 
                                storage=hist.storage.Weight())
    _hist.fill(values, weight=weights)
    _normed_hist = normalize_hist(_hist)

    return _hist, _normed_hist

def make_empty_hist(bins):
    _hist = Hist(hist.axis.Regular(bins=len(bins)-1, start=bins[0], stop=bins[-1], overflow=True, underflow=True), 
                                storage=hist.storage.Weight())
    return _hist


def digitize_pd(pd_input, reweight='event_weight', is_MC = True):

    HistMap_unumpy = {}
    if is_MC:
        _label_jettype = label_jettype
    else:
        _label_jettype = label_jettype_data
    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        pt_input_idx = pd_input['pt_idx'] == pt_idx
        pd_input_at_pt = pd_input[pt_input_idx]

        for leadingtype in label_leadingtype:
            leadingtype_idx = pd_input_at_pt['is_leading'].isin(is_leading_map[leadingtype])
            pd_input_at_pt_leadingtype = pd_input_at_pt[leadingtype_idx]
            
            for eta_region in label_etaregion: 
                etaregion_idx = pd_input_at_pt_leadingtype['is_forward'].isin(is_forward_map[eta_region])
                pd_input_at_pt_leadingtype_etaregion = pd_input_at_pt_leadingtype[etaregion_idx]
                
                for jettype in _label_jettype:
                    type_idx = pd_input_at_pt_leadingtype_etaregion['jet_PartonTruthLabelID'].isin(label_jettype_map[jettype])
                    pd_input_at_pt_leadingtype_etaregion_jettype = pd_input_at_pt_leadingtype_etaregion[type_idx]
                    for var in label_var:
                        key = f"{pt}_{leadingtype}_{eta_region}_{jettype}_{var}"
                        bin_var = HistBins[var] # Get rid of meaningless naming conversions! 

                        # TODO: Can change the format from unumpy to hist. Now just to test the plotting code. 
                        if len(pd_input_at_pt_leadingtype_etaregion_jettype) == 0: ## for subset, if len==0, give it an empty unumpy array
                            # HistMap_unumpy[key] = unumpy.uarray(np.zeros(len(bin_var)-1), np.zeros(len(bin_var)-1))
                            _hist = make_empty_hist(bins=bin_var)

                        else:
                            _hist, _ = make_hist(values=pd_input_at_pt_leadingtype_etaregion_jettype[var],
                                                bins=bin_var, weights=pd_input_at_pt_leadingtype_etaregion_jettype[reweight])
                            # HistMap_unumpy[key] = unumpy.uarray(_hist.values(), np.sqrt(_hist.variances()))
                        HistMap_unumpy[key] = _hist
                            
    return HistMap_unumpy

def attach_reweight_factor(pd_input, reweight_factor):
    reweighting_vars = ['jet_nTracks', 'GBDT_newScore'] 
    for reweighting_var in reweighting_vars:
        pd_input[f'{reweighting_var}_quark_reweighting_weights'] = pd_input['event_weight'].copy()
        pd_input[f'{reweighting_var}_gluon_reweighting_weights'] = pd_input['event_weight'].copy()


    #### reweight_factor[pt][var]['quark_factor']
    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        # forward_idx = (pd_input['pt_idx'] == pt_idx) & (pd_input['is_forward'] == 1) # No need to change forward 
        central_mask = (pd_input['pt_idx'] == pt_idx) & (pd_input['is_forward'] == 0)
        # pd_input_at_pt_forward = pd_input.loc[(pd_input['pt_idx'] == pt_idx) & (pd_input['is_forward'] == 1)]
        # pd_input_at_pt_central = pd_input.loc[(pd_input['pt_idx'] == pt_idx) & (pd_input['is_forward'] == 0)]
        if central_mask.sum() == 0: ## This is crutial doing subset. mod = [] and then df.iloc will run into error 
            # by central_mask, its type is True/False Series 
            continue

        for reweighting_var in reweighting_vars:
            bin_var = HistBins[reweighting_var]

            quark_factor_col = f'{reweighting_var}_quark_reweighting_weights'
            gluon_factor_col = f'{reweighting_var}_gluon_reweighting_weights'

            quark_factor = reweight_factor[pt][reweighting_var]['quark_factor']
            gluon_factor = reweight_factor[pt][reweighting_var]['gluon_factor']
            
            var_idx = np.digitize(x=pd_input.loc[central_mask, reweighting_var], bins=bin_var) - 1  # Binned feature distribution 
            if np.max(var_idx) == bin_var[-1]: # out of boundary, e.g. ntrk = 70  
                quark_factor = np.append(quark_factor, 1)
                gluon_factor = np.append(gluon_factor, 1)

            pd_input.loc[central_mask, quark_factor_col] *= quark_factor[var_idx]
            pd_input.loc[central_mask, gluon_factor_col] *= gluon_factor[var_idx]
            # for i, score in enumerate(bin_var[:-1]): # Loop over the bins 
            #     mod_idx = np.where(var_idx == i)[0]
            #     pd_input_at_pt_central.iloc[mod_idx, quark_factor_idx] *= quark_factor[i]
            #     pd_input_at_pt_central.iloc[mod_idx, gluon_factor_idx] *= gluon_factor[i]
            
        # reweighted_sample.append(pd_input_at_pt_forward)
        # reweighted_sample.append(pd_input_at_pt_central)                
    
    return pd_input

def safe_array_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(numerator, denominator)
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
    return ratio

def calculate_reweight_factor(values_forward, values_central, weights_forward, weights_central, bins): 
    _, normed_hist_forward = make_hist(values_forward, bins=bins, weights=weights_forward)
    _, normed_hist_central = make_hist(values_central, bins=bins, weights=weights_central)

    return safe_array_divide(numerator=normed_hist_forward.values(), denominator=normed_hist_central.values())

def construct_etaregion(hist_dict, bins, eta_region, pt, var, jet_type = 'Quark'): 
    # key = f"{pt}_{leadingtype}_{eta_region}_{jettype}_{var}"
    # 500_LeadingJet_Forward_Gluon_pt 
    # One can use regular expression     
    if jet_type == 'Quark':
        jet_types = ['Quark', 'C_Quark', 'B_Quark']
    elif jet_type == 'Gluon':
        jet_types = ['Gluon']
    else:
        raise Exception(f'Check your argument jet_type. Current input {jet_type}.')

    return_hist = make_empty_hist(bins=bins) # Forward or central 
    for leadingtype in label_leadingtype:
        for jet_type in jet_types:
            return_hist += hist_dict[f"{pt}_{leadingtype}_{eta_region}_{jet_type}_{var}"]
    
    return normalize_hist(return_hist) 

    

def get_reweight_factor_hist(MC_hists_dict, if_need_merge = True):
    reweight_factor = {}
    if if_need_merge:
        # This assumes MC_hists_dict['pythiaA'][file1][500...]
        # Merge ADE first 
        MC_hist_dicts_keys = [*MC_hists_dict] # MC_hist_dicts_keys ['A', 'D', 'E'] 
        # Merge A first
        merged_ADE_dict = MC_hists_dict[MC_hist_dicts_keys[0]][0]
        for hists_file in MC_hists_dict[MC_hist_dicts_keys[0]][1:]:
            for k in merged_ADE_dict:
                merged_ADE_dict[k] += hists_file[k]

        # Merge D, E 
        for other_key in MC_hist_dicts_keys[1:]: # ['D', 'E']
            for hists_file in MC_hists_dict[other_key]:
                for k in merged_ADE_dict: 
                    merged_ADE_dict[k] += hists_file[k]
        MC_hists_dict = merged_ADE_dict

    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        reweight_factor[pt] = {}

        for var in reweighting_vars:
            bin_var = HistBins[var]

            forward_quark = construct_etaregion(hist_dict=MC_hists_dict, bins=bin_var, 
                                                eta_region=label_etaregion[0], jet_type='Quark', pt=pt, var=var)
            central_quark = construct_etaregion(hist_dict=MC_hists_dict, bins=bin_var, 
                                                eta_region=label_etaregion[1], jet_type='Quark', pt=pt, var=var)
            forward_gluon = construct_etaregion(hist_dict=MC_hists_dict, bins=bin_var, 
                                                eta_region=label_etaregion[0], jet_type='Gluon', pt=pt, var=var)
            central_gluon = construct_etaregion(hist_dict=MC_hists_dict, bins=bin_var, 
                                                eta_region=label_etaregion[1], jet_type='Gluon', pt=pt, var=var)
            reweight_factor[pt][var]={
                'quark_factor': safe_array_divide(numerator=forward_quark.values(), denominator=central_quark.values()),
                'gluon_factor': safe_array_divide(numerator=forward_gluon.values(), denominator=central_gluon.values())
            }                                                                          

    return reweight_factor

def get_reweight_factor_pd(reweighting_input):
    reweight_factor = {}
    label_pt_bin = np.array([500, 600, 800, 1000, 1200, 1500, 2000])

    for pt_idx, pt in enumerate(label_pt_bin[:-1]):
        reweight_factor[pt] = {}
        reweighting_input_at_pt = reweighting_input[reweighting_input['pt_idx'] == pt_idx]

        forward_quark_at_pt = reweighting_input_at_pt.loc[(reweighting_input_at_pt['target'] == 0) & 
                                                        (reweighting_input_at_pt['is_forward'] == 1)]
        central_quark_at_pt = reweighting_input_at_pt.loc[(reweighting_input_at_pt['target'] == 0) & 
                                                        (reweighting_input_at_pt['is_forward'] == 0)]
        forward_gluon_at_pt = reweighting_input_at_pt.loc[(reweighting_input_at_pt['target'] == 1) & 
                                                        (reweighting_input_at_pt['is_forward'] == 1)]
        central_gluon_at_pt = reweighting_input_at_pt.loc[(reweighting_input_at_pt['target'] == 1) & 
                                                        (reweighting_input_at_pt['is_forward'] == 0)]   

        for var in reweighting_vars:
            bin_var = HistBins[var]

            quark_factor = calculate_reweight_factor(values_forward=forward_quark_at_pt[var], values_central=central_quark_at_pt[var],
                                                     weights_forward=forward_quark_at_pt['event_weight'], weights_central=central_quark_at_pt['event_weight'],
                                                     bins=bin_var)

            gluon_factor = calculate_reweight_factor(values_forward=forward_gluon_at_pt[var], values_central=central_gluon_at_pt[var],
                                                     weights_forward=forward_gluon_at_pt['event_weight'], weights_central=central_gluon_at_pt['event_weight'],
                                                     bins=bin_var)
            
            reweight_factor[pt][var]={
                'quark_factor' : quark_factor, 
                'gluon_factor' : gluon_factor, 
            }

    return reweight_factor

def convert_unumpy2hist(unumpy_array, bins):
    _hist = Hist(hist.axis.Regular(bins=len(bins)-1, start=bins[0], stop=bins[-1], 
                                overflow=True, underflow=True), 
                                storage=hist.storage.Weight())
    # assert np.sum(unumpy.nominal_values(unumpy_array)) == np.sum(unumpy.std_devs(unumpy_array) **2 ) 
    # print(np.sum(unumpy.std_devs(unumpy_array) **2 ) )
    _hist[:] = np.hstack((unumpy.nominal_values(unumpy_array)[:,None], unumpy.std_devs(unumpy_array)[:,None]**2))

    return _hist

def convert_hist2unumpy(_hist):
    if isinstance(_hist, hist.hist.Hist):
        _unumpy = unumpy.uarray(_hist.values(),
                                np.sqrt(_hist.variances()))
        return _unumpy 
    else:
        raise Exception(f"check the input type {type(_hist)}")
    
###########################################################
######################Analysis steps#######################
###########################################################



###########################################################
######################Uncertainties########################
###########################################################

trk_eff_uncertainties = ['jet_nTracks_sys_eff', 'jet_nTracks_sys_fake']

JESJER_uncertainties = \
['syst_JET_BJES_Response__1up',
 'syst_JET_BJES_Response__1down',
 'syst_JET_EffectiveNP_Detector1__1up',
 'syst_JET_EffectiveNP_Detector1__1down',
 'syst_JET_EffectiveNP_Detector2__1up',
 'syst_JET_EffectiveNP_Detector2__1down',
 'syst_JET_EffectiveNP_Mixed1__1up',
 'syst_JET_EffectiveNP_Mixed1__1down',
 'syst_JET_EffectiveNP_Mixed2__1up',
 'syst_JET_EffectiveNP_Mixed2__1down',
 'syst_JET_EffectiveNP_Mixed3__1up',
 'syst_JET_EffectiveNP_Mixed3__1down',
 'syst_JET_EffectiveNP_Modelling1__1up',
 'syst_JET_EffectiveNP_Modelling1__1down',
 'syst_JET_EffectiveNP_Modelling2__1up',
 'syst_JET_EffectiveNP_Modelling2__1down',
 'syst_JET_EffectiveNP_Modelling3__1up',
 'syst_JET_EffectiveNP_Modelling3__1down',
 'syst_JET_EffectiveNP_Modelling4__1up',
 'syst_JET_EffectiveNP_Modelling4__1down',
 'syst_JET_EffectiveNP_R10_1__1up',
 'syst_JET_EffectiveNP_R10_1__1down',
 'syst_JET_EffectiveNP_R10_2__1up',
 'syst_JET_EffectiveNP_R10_2__1down',
 'syst_JET_EffectiveNP_R10_3__1up',
 'syst_JET_EffectiveNP_R10_3__1down',
 'syst_JET_EffectiveNP_R10_4__1up',
 'syst_JET_EffectiveNP_R10_4__1down',
 'syst_JET_EffectiveNP_R10_5__1up',
 'syst_JET_EffectiveNP_R10_5__1down',
 'syst_JET_EffectiveNP_R10_6restTerm__1up',
 'syst_JET_EffectiveNP_R10_6restTerm__1down',
 'syst_JET_EffectiveNP_Statistical1__1up',
 'syst_JET_EffectiveNP_Statistical1__1down',
 'syst_JET_EffectiveNP_Statistical2__1up',
 'syst_JET_EffectiveNP_Statistical2__1down',
 'syst_JET_EffectiveNP_Statistical3__1up',
 'syst_JET_EffectiveNP_Statistical3__1down',
 'syst_JET_EffectiveNP_Statistical4__1up',
 'syst_JET_EffectiveNP_Statistical4__1down',
 'syst_JET_EffectiveNP_Statistical5__1up',
 'syst_JET_EffectiveNP_Statistical5__1down',
 'syst_JET_EffectiveNP_Statistical6__1up',
 'syst_JET_EffectiveNP_Statistical6__1down',
 'syst_JET_EtaIntercalibration_Modelling__1up',
 'syst_JET_EtaIntercalibration_Modelling__1down',
 'syst_JET_EtaIntercalibration_NonClosure_2018data__1up',
 'syst_JET_EtaIntercalibration_NonClosure_2018data__1down',
 'syst_JET_EtaIntercalibration_NonClosure_highE__1up',
 'syst_JET_EtaIntercalibration_NonClosure_highE__1down',
 'syst_JET_EtaIntercalibration_NonClosure_negEta__1up',
 'syst_JET_EtaIntercalibration_NonClosure_negEta__1down',
 'syst_JET_EtaIntercalibration_NonClosure_posEta__1up',
 'syst_JET_EtaIntercalibration_NonClosure_posEta__1down',
 'syst_JET_EtaIntercalibration_R10_TotalStat__1up',
 'syst_JET_EtaIntercalibration_R10_TotalStat__1down',
 'syst_JET_EtaIntercalibration_TotalStat__1up',
 'syst_JET_EtaIntercalibration_TotalStat__1down',
 'syst_JET_Flavor_Composition__1up',
 'syst_JET_Flavor_Composition__1down',
 'syst_JET_Flavor_Response__1up',
 'syst_JET_Flavor_Response__1down',
 'syst_JET_JER_DataVsMC_MC16__1up',
 'syst_JET_JER_DataVsMC_MC16__1down',
 'syst_JET_JER_EffectiveNP_1__1up',
 'syst_JET_JER_EffectiveNP_1__1down',
 'syst_JET_JER_EffectiveNP_10__1up',
 'syst_JET_JER_EffectiveNP_10__1down',
 'syst_JET_JER_EffectiveNP_11__1up',
 'syst_JET_JER_EffectiveNP_11__1down',
 'syst_JET_JER_EffectiveNP_12restTerm__1up',
 'syst_JET_JER_EffectiveNP_12restTerm__1down',
 'syst_JET_JER_EffectiveNP_2__1up',
 'syst_JET_JER_EffectiveNP_2__1down',
 'syst_JET_JER_EffectiveNP_3__1up',
 'syst_JET_JER_EffectiveNP_3__1down',
 'syst_JET_JER_EffectiveNP_4__1up',
 'syst_JET_JER_EffectiveNP_4__1down',
 'syst_JET_JER_EffectiveNP_5__1up',
 'syst_JET_JER_EffectiveNP_5__1down',
 'syst_JET_JER_EffectiveNP_6__1up',
 'syst_JET_JER_EffectiveNP_6__1down',
 'syst_JET_JER_EffectiveNP_7__1up',
 'syst_JET_JER_EffectiveNP_7__1down',
 'syst_JET_JER_EffectiveNP_8__1up',
 'syst_JET_JER_EffectiveNP_8__1down',
 'syst_JET_JER_EffectiveNP_9__1up',
 'syst_JET_JER_EffectiveNP_9__1down',
 'syst_JET_JvtEfficiency__1down',
 'syst_JET_JvtEfficiency__1up',
 'syst_JET_LargeR_TopologyUncertainty_V__1up',
 'syst_JET_LargeR_TopologyUncertainty_V__1down',
 'syst_JET_LargeR_TopologyUncertainty_top__1up',
 'syst_JET_LargeR_TopologyUncertainty_top__1down',
 'syst_JET_Pileup_OffsetMu__1up',
 'syst_JET_Pileup_OffsetMu__1down',
 'syst_JET_Pileup_OffsetNPV__1up',
 'syst_JET_Pileup_OffsetNPV__1down',
 'syst_JET_Pileup_PtTerm__1up',
 'syst_JET_Pileup_PtTerm__1down',
 'syst_JET_Pileup_RhoTopology__1up',
 'syst_JET_Pileup_RhoTopology__1down',
 'syst_JET_PunchThrough_MC16__1up',
 'syst_JET_PunchThrough_MC16__1down',
 'syst_JET_SingleParticle_HighPt__1up',
 'syst_JET_SingleParticle_HighPt__1down']

pdf_weight_uncertainties = list(map(str, list(np.arange(1, 101, 1))))
scale_variation_uncertainties = list(map(str, list(np.arange(1, 27, 1))))
parton_shower_uncertainties = ["herwigangle", "herwigdipole"]
hadronization_uncertainties = ["sherpa", "sherpalund"]
matrix_element_uncertainties = ["powhegpythia"]

all_systs_subtypes = trk_eff_uncertainties + JESJER_uncertainties + pdf_weight_uncertainties + \
                     scale_variation_uncertainties + parton_shower_uncertainties + hadronization_uncertainties +\
                     matrix_element_uncertainties 
