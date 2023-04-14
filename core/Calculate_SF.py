import argparse
from genericpath import exists
import uproot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path
import pickle
import joblib
from .utils import HistBins, label_pt_bin
from uncertainties import ufloat, unumpy
import hist 
from hist import Hist
import logging
import atlas_mpl_style as ampl
ampl.use_atlas_style(usetex=False)

#label_var = ['jet_nTracks', 'GBDT_newScore']
label_var = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'jet_trackBDT', 'GBDT_newScore']
label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]

def Construct_unumpy(HistMap_unumpy, n_bins, sampletype):
    ## Construct data-like MC 
    Forward_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 

    for k, v in HistMap_unumpy.items():
        if k.__contains__('Forward'):
            Forward_unumpy += v
        elif k.__contains__('Central'):
            Central_unumpy += v

    if sampletype == "Data":
        return Forward_unumpy, Central_unumpy

    ## Construct pure Quark vs Gluon 
    Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    for k, v in HistMap_unumpy.items():
        if k.__contains__('Quark'):
            Quark_unumpy += v
        elif k.__contains__('Gluon'):
            Gluon_unumpy += v

    Forward_Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Forward_Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_Quark_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 
    Central_Gluon_unumpy = unumpy.uarray(np.zeros((n_bins)), np.zeros((n_bins))) 

    for k, v in HistMap_unumpy.items():
        if k.__contains__('Quark') and k.__contains__('Forward'):
            Forward_Quark_unumpy += v
        elif k.__contains__('Gluon') and k.__contains__('Forward'):
            Forward_Gluon_unumpy += v
        elif k.__contains__('Quark') and k.__contains__('Central'):
            Central_Quark_unumpy += v
        elif k.__contains__('Gluon') and k.__contains__('Central'):
            Central_Gluon_unumpy += v
    return Forward_unumpy, Central_unumpy, Quark_unumpy, Gluon_unumpy, Forward_Quark_unumpy, Forward_Gluon_unumpy, Central_Quark_unumpy, Central_Gluon_unumpy

def Calcu_Frac(Forward_Quark, Central_Quark, Forward, Central):
    try:
        frac_Forward_Quark = np.sum(Forward_Quark) / np.sum(Forward)
        frac_Central_Quark = np.sum(Central_Quark) / np.sum(Central)
    except RuntimeWarning:
        print(Forward)
        print(Central)

    frac_Forward_Gluon = 1 - frac_Forward_Quark
    frac_Central_Gluon = 1 - frac_Central_Quark

    f = np.array([[frac_Forward_Quark,  frac_Forward_Gluon], [frac_Central_Quark, frac_Central_Gluon]])

    return f, np.linalg.inv(f)

def Calcu_Frac_unumpy(Forward_Quark, Central_Quark, Forward, Central):
    try:
        frac_Forward_Quark = np.sum(Forward_Quark) / np.sum(Forward)
        frac_Central_Quark = np.sum(Central_Quark) / np.sum(Central)
    except RuntimeWarning:
        print(Forward)
        print(Central)

    frac_Forward_Gluon = 1 - frac_Forward_Quark
    frac_Central_Gluon = 1 - frac_Central_Quark

    f = np.array([[frac_Forward_Quark,  frac_Forward_Gluon], [frac_Central_Quark, frac_Central_Gluon]])

    return f, unumpy.ulinalg.inv(f)

def Normalize_unumpy(array_unumpy, bin_width=1.0):
    area = np.sum(unumpy.nominal_values(array_unumpy)) * bin_width
    return array_unumpy / area

def safe_array_divide_unumpy(numerator, denominator):
    if 0 in unumpy.nominal_values(denominator):
        _denominator_nominal_values = unumpy.nominal_values(denominator)
        _denominator_std_devs = unumpy.std_devs(denominator)
        zero_idx = np.where(_denominator_nominal_values==0)[0]
        _denominator_nominal_values[zero_idx] = np.inf
        _denominator_std_devs[zero_idx] = 0 
        _denominator = unumpy.uarray(_denominator_nominal_values, _denominator_std_devs)

        ratio = np.true_divide(numerator, _denominator) 
        # raise Warning(f"0 exists in the denominator for unumpy, check it!")
    else:
        ratio = np.true_divide(numerator, denominator)        
    return ratio

Map_var_title = {
    "jet_pt": "$p_{T}$",
    "jet_nTracks": "$N_{trk}$",
    "jet_trackBDT": "old BDT",
    "jet_eta": "$\eta$",
    "jet_trackC1": "$C_{1}$",
    "jet_trackWidth": "$W_{trk}$",
    "GBDT_newScore": "GBDT"
}

def Read_Histogram_Root(file, sampletype="MC", code_version="new", reweighting_var=None, reweighting_factor="none"):
    """A general func to read the contents of a root file. In future we'll discard the root format.

    Args:
        file (str): the path to the file you want to read
        sampletype (str, optional): MC or Data. Jet type not known in Data. Defaults to "MC".
        code_version (str, optional): new or old. new is being developed. Defaults to "new".
        reweighting_var (str, optional): ntrk or bdt. Defaults to None.
        reweighting_factor (str, optional): quark or gluon. Defaults to "none".

    Returns:
        (Dict, Dict): Return HistMap and HistMap_Error. 
    """
    # defile which TDirectory to look at based on {reweighting_var}_{reweighting_factor}
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    if sampletype== "MC":
        label_jettype = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
    elif sampletype == "Data":
        label_jettype = ["Data"]
    
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    label_var = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'jet_trackBDT', 'GBDT_newScore']

    HistMap = {}
    HistMap_Error = {}
    HistMap_unumpy = {}

    if code_version=="new":
        if reweighting_factor == "none":
            TDirectory_name = reweighting_map[reweighting_factor]
        else:
            TDirectory_name = f"{reweighting_var}_" + reweighting_map[reweighting_factor]

        file = uproot.open(file)[TDirectory_name]
    elif code_version=="old":
        file = uproot.open(file)
    
    avail_keys = [*file.keys()]
    for pt in label_ptrange[:-1]:
        for leadingtype in label_leadingtype:
            for eta_region in label_etaregion: 
                for var in label_var:
                    for jettype in label_jettype:

                        key = f"{pt}_{leadingtype}_{eta_region}_{jettype}_{var}"
                        if (key in avail_keys) or (key+";1" in avail_keys):
                            HistMap[key] = file[key].to_numpy()[0]
                            HistMap_Error[key] = file[f"{key}_err"].to_numpy()[0] # No more suffix '_err' in HistMap_Error

    
    for key, value in HistMap.items():
        HistMap_unumpy[key] = unumpy.uarray(value, np.sqrt(HistMap_Error[key]))
    return HistMap, HistMap_Error, HistMap_unumpy

def convert_hist2unumpy(_hist):
    if isinstance(_hist, hist.hist.Hist):
        _unumpy = unumpy.uarray(_hist.values(),
                                np.sqrt(_hist.variances()))
        return _unumpy 
    else:
        raise Exception(f"check the input type {type(_hist)}")

def convert_histdict2unumpy(_hist_dict):
    new_dict = {}
    for k, v in _hist_dict.items():
        new_dict[k] = convert_hist2unumpy(v)
    
    return new_dict

def Read_Histogram_pkl(file, reweighting_var, reweighting_factor):
    weight_option_map ={
        'none': 'event_weight',
        'quark': 'quark_reweighting_weights',
        'gluon': 'gluon_reweighting_weights'
    }
    var_map = {
        "ntrk":'jet_nTracks',
        "bdt":'jet_trackBDT',
        "newBDT":'GBDT_newScore'
    }

    hists = joblib.load(file)
    if reweighting_factor == 'none':
        return_hist = convert_histdict2unumpy(hists[f"{weight_option_map[reweighting_factor]}"])
    else:
        return_hist = convert_histdict2unumpy(hists[f"{var_map[reweighting_var]}_{weight_option_map[reweighting_factor]}"])
    return return_hist

def Extract(HistMap_MC_unumpy, HistMap_Data_unumpy):
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    #label_var = ['jet_nTracks', 'GBDT_newScore']
    label_var = ['jet_pt', 'jet_eta', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'jet_trackBDT', 'GBDT_newScore']

    # label_var = ['ntrk', 'bdt']
    # label_var = ['ntrk']
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    label_type = ["Gluon", "Quark", "B_Quark", "C_Quark"]

    # HistMap_MC, HistMap_Error_MC, HistMap_MC_unumpy = Read_Histogram_Root(input_mc_path, sampletype="MC", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")
    # HistMap_Data, HistMap_Error_Data, HistMap_Data_unumpy = Read_Histogram_Root(input_data_path, sampletype="Data", code_version="new", reweighting_var=reweighting_var, reweighting_factor="quark")
    Extraction_Results = {}
    for var in label_var:
        Extraction_Results[var] = {}
        for l_pt in label_ptrange[:-1]:

            sel_HistMap_MC_unumpy = {}
            sel_HistMap_Data_unumpy = {}

            for i, l_leadingtype  in enumerate(label_leadingtype):
                for j, l_etaregion in enumerate(label_etaregion):
                    key_data = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + "Data" + "_" + var
                    sel_HistMap_Data_unumpy[key_data] = HistMap_Data_unumpy[key_data]

                    for k, l_type in enumerate(label_type):
                        key_mc = str(l_pt) + "_" + l_leadingtype + "_" + l_etaregion + "_" + l_type + "_" + var
                        sel_HistMap_MC_unumpy[key_mc] = HistMap_MC_unumpy[key_mc]
            # The following two lines left for check the mannual calclulation 
            # Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct(HistMap=sel_HistMap_MC_Manual, HistMap_Error=sel_HistMap_Error_MC_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="MC")
            # Forward, Central = Construct(HistMap=sel_HistMap_Data_Manual, HistMap_Error=sel_HistMap_Error_Data_Manual, n_bins = len(GetHistBin(var)) - 1, sampletype="Data")

            Forward, Central, Quark, Gluon, Forward_Quark, Forward_Gluon, Central_Quark, Central_Gluon  = Construct_unumpy(HistMap_unumpy=sel_HistMap_MC_unumpy, n_bins = len(HistBins[var]) - 1, sampletype="MC")
            Forward_Data, Central_Data = Construct_unumpy(HistMap_unumpy=sel_HistMap_Data_unumpy, n_bins = len(HistBins[var]) - 1, sampletype="Data")
           
            # f, f_inv = Calcu_Frac_unumpy(Forward_Quark, Central_Quark, Forward, Central)
            f, f_inv = Calcu_Frac(unumpy.nominal_values(Forward_Quark), unumpy.nominal_values(Central_Quark), unumpy.nominal_values(Forward), unumpy.nominal_values(Central))
            # normalize 
            ## Truth
            p_Quark = Normalize_unumpy(Quark)
            p_Gluon = Normalize_unumpy(Gluon)

            p_Forward_Quark = Normalize_unumpy(Forward_Quark)
            p_Central_Quark = Normalize_unumpy(Central_Quark)
            p_Forward_Gluon = Normalize_unumpy(Forward_Gluon)
            p_Central_Gluon = Normalize_unumpy(Central_Gluon)

            p_Forward = Normalize_unumpy(Forward)
            p_Central = Normalize_unumpy(Central)
            p_Forward_Data = Normalize_unumpy(Forward_Data)
            p_Central_Data = Normalize_unumpy(Central_Data)
            
            extract_p_Quark = f_inv[0][0] * p_Forward + f_inv[0][1]* p_Central 
            extract_p_Gluon = f_inv[1][0] * p_Forward + f_inv[1][1]* p_Central 

            extract_p_Quark_Data = f_inv[0][0] * p_Forward_Data + f_inv[0][1]* p_Central_Data 
            extract_p_Gluon_Data = f_inv[1][0] * p_Forward_Data + f_inv[1][1]* p_Central_Data 
        
            Extraction_Results[var][l_pt] = {
                "Forward_MC": Forward,
                "Central_MC": Central,
                "Forward_Data": Forward_Data,
                "Central_Data": Central_Data,
                "Quark":Quark,
                "Gluon":Gluon,
                "Forward_Quark":Forward_Quark,
                "Central_Quark":Central_Quark,
                "Forward_Gluon":Forward_Gluon,
                "Central_Gluon":Central_Gluon,
                "f": f,
                "f_inv": f_inv,
                "p_Quark": p_Quark,
                "p_Gluon": p_Gluon,
                "p_Forward_Quark": p_Forward_Quark,
                "p_Central_Quark": p_Central_Quark,
                "p_Forward_Gluon": p_Forward_Gluon,
                "p_Central_Gluon": p_Central_Gluon,
                "extract_p_Quark_MC": extract_p_Quark,
                "extract_p_Gluon_MC": extract_p_Gluon,
                "extract_p_Quark_Data": extract_p_Quark_Data,
                "extract_p_Gluon_Data": extract_p_Gluon_Data
            }

    return Extraction_Results

def cal_sum_unumpy(Read_HistMap_MC):
    """For MC sample only, this func is to calculate the sum of each type. 

    Args:
        Read_HistMap (Dict): the output of Read_Histogram by JetType

    Returns:
        np.array: sum of different types 
    """
    MC_jet_types = ['C_Quark', 'B_Quark', 'Gluon', 'Quark']

    MC_jets = []
    MC_jets.append(unumpy.uarray(nominal_values=np.zeros(60), std_devs=np.zeros(60)))
    for MC_jet_type in MC_jet_types:
        MC_jets.append(Read_HistMap_MC[MC_jet_type])

    MC_jets = np.array(MC_jets)

    cumsum_MC_jets = np.cumsum(MC_jets, axis = 0)
    assert np.allclose(unumpy.nominal_values(Read_HistMap_MC[MC_jet_types[0]]),
                       unumpy.nominal_values(cumsum_MC_jets[1]))
    return cumsum_MC_jets

def Plot_Pt_Spectrum(HistMap_MC_unumpy, HistMap_Data_unumpy, output_path, reweighting_var, reweighting_option, period='ADE'):
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_leadingtype = ["LeadingJet", "SubLeadingJet"]
    label_etaregion = ["Forward", "Central"]
    # label_jettype_MC = ["Quark", "Gluon", "B_Quark", "C_Quark", "Other"]
    label_jettype_MC = ["Quark", "Gluon", "B_Quark", "C_Quark"] # No other(partonID==-1) in new workflow 
    label_jettype_Data = ["Data"]
    label_jettype = [label_jettype_MC, label_jettype_Data]
    n_bins_var = [60, 50, 60, 60, 60, 60]

    Read_HistMap_MC = {}
    Read_HistMap_Data = {}
    Read_HistMap = [Read_HistMap_MC, Read_HistMap_Data]
    HistMap_unumpy = [HistMap_MC_unumpy, HistMap_Data_unumpy]

    for i_sample, Read_HistMap_sample in enumerate(Read_HistMap):
        HistMap_unumpy_sample = HistMap_unumpy[i_sample]
        for i_leading, l_leadingtype in enumerate(label_leadingtype):
            Read_HistMap_sample[l_leadingtype] = {}
            label_jettype_sample = label_jettype[i_sample]
            for i, jettype in enumerate(label_jettype_sample):
                Read_HistMap_sample[l_leadingtype][jettype] = unumpy.uarray(np.zeros(n_bins_var[0]), np.zeros(n_bins_var[0]))
                for pt in label_ptrange[:-1]:
                    for eta_region in label_etaregion: 
                        key_name = f"{pt}_{l_leadingtype}_{eta_region}_{jettype}_{label_var[0]}"
                        Read_HistMap_sample[l_leadingtype][jettype] += HistMap_unumpy_sample[key_name]
        
    assert sorted(label_jettype_MC) == sorted([*Read_HistMap[0][label_leadingtype[0]]])
    assert sorted(label_jettype_Data) == sorted([*Read_HistMap[1][label_leadingtype[0]]])
    ##
    # Read_HistMap_MC["LeadingJet"]["C_Quark"]
    #### Plot here 
    MC_jet_types = ['C_Quark', 'B_Quark', 'Gluon', 'Quark']

    for i_leading, l_leadingtype in enumerate(label_leadingtype): 
        cumsum_MC_jets = cal_sum_unumpy(Read_HistMap_MC=Read_HistMap_MC[l_leadingtype])
        fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
        custom_bins = np.linspace(0, 2000, 61)
        pt_bin_centers = 1/2 * (custom_bins[:-1] + custom_bins[1:])

        for i in range(0, len(cumsum_MC_jets)-1):
            ax.fill_between(pt_bin_centers, unumpy.nominal_values(cumsum_MC_jets[i]), unumpy.nominal_values(cumsum_MC_jets[i+1]), 
                            label = MC_jet_types[i], step='mid')
                            #label = MC_jet_types[i]+ f", num:{np.sum(unumpy.nominal_values(Read_HistMap_MC[l_leadingtype][MC_jet_types[i]])):.2e}", step='mid')

        total_jet_MC = unumpy.nominal_values(cumsum_MC_jets[-1])
        total_jet_Data = unumpy.nominal_values(Read_HistMap_Data[l_leadingtype]['Data'])
        total_jet_error_MC = unumpy.std_devs(cumsum_MC_jets[-1])
        total_jet_error_Data = unumpy.std_devs(Read_HistMap_Data[l_leadingtype]['Data'])

        # # ax.stairs(values=cumsum_MC_jets[-1], edges=custom_bins, label = "Total MC"+ f"num. {np.sum(cumsum_MC_jets[-1]):.2f}" )
        ax.errorbar(x = pt_bin_centers, y = total_jet_MC, yerr = total_jet_error_MC, drawstyle = 'steps-mid', label = "Total MC")
        ax.errorbar(x = pt_bin_centers, y = total_jet_Data, yerr = total_jet_error_Data, drawstyle = 'steps-mid', color= "black", linestyle='', marker= "o", markersize=10, label = "Data") 
        #ax.errorbar(x = pt_bin_centers, y = total_jet_MC, yerr = total_jet_error_MC, drawstyle = 'steps-mid', label = "Total MC"+ f", num:{np.sum(total_jet_MC):.2e}")
        #ax.errorbar(x = pt_bin_centers, y = total_jet_Data, yerr = total_jet_error_Data, drawstyle = 'steps-mid', color= "black", linestyle='', marker= "o", markersize=10, label = "Data" + f", num:{np.sum(total_jet_Data):.2e}")

        ampl.plot.draw_atlas_label(0.1, 0.85, ax=ax, energy="13 TeV",lumi=139)
        ax.set_yscale('log')
        ax.set_xlim(500, 2000)
        ax.set_ylim(1e3,1e8)
#        ax.set_title(f'MC16{period} {l_leadingtype}' +  ' Jet $p_{T}$ Spectrum Component')
        #ax.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
        ampl.plot.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
        ax.set_ylabel('Number of Events')

        ax.legend()

        ratio = safe_array_divide_unumpy(Read_HistMap_Data[l_leadingtype]['Data'], cumsum_MC_jets[-1])
        ax1.errorbar(pt_bin_centers, y=unumpy.nominal_values(ratio), yerr = unumpy.std_devs(ratio), color= "black", drawstyle = 'steps-mid')
        ax1.hlines(y = 1, xmin = 500, xmax = 2000, color = 'gray', linestyle = '--')
        ax1.set_ylabel("Data/MC")
        ax1.set_ylim(0.7, 1.3)
        ax1.legend()

        output_path_new = output_path / period / "Pt_spectrum" / f"{reweighting_var}_{reweighting_option}" 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True, exist_ok =True)
        fig.savefig(output_path_new/f'pt_MC16{period}_{l_leadingtype}.pdf')
        plt.close()

def _Plot_ROC(p_Quark_unumpy, p_Gluon_unumpy, l_ptrange, etaregion):
    label_var = ['jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'GBDT_newScore']

    fig, ax0 = plt.subplots()
    for i_var, l_var in enumerate(label_var):
        p_Quark = unumpy.nominal_values(p_Quark_unumpy[l_var])
        p_Gluon = unumpy.nominal_values(p_Gluon_unumpy[l_var])

        var_bins = HistBins[l_var]
        n_cut = len(var_bins)-1
        quark_effs = np.zeros(n_cut)
        gluon_rejs = np.zeros(n_cut)

        for cut_idx in range(n_cut):
            TP = np.sum(p_Quark[:cut_idx])
            TN = np.sum(p_Gluon[cut_idx:])
            quark_effs[cut_idx] = TP ## After normalization 
            gluon_rejs[cut_idx] = TN
        # auc = 
        ax0.plot(quark_effs, gluon_rejs, label = f"{Map_var_title[l_var]}")

    #ax0.set_title(f"ROC for truth q/g at {l_ptrange} GeV, {etaregion}")
    ax0.set_xlabel("Quark Efficiency")
    ax0.set_ylabel("1 - Gluon Rejection")
    
    ax0.set_xticks(np.linspace(0, 1, 11))
    ax0.set_xlim(0,1)
    ax0.set_yticks(np.linspace(0, 1, 21))
    ax0.set_ylim(0,1)
    ax0.legend()
    ax0.grid()
    ampl.plot.draw_atlas_label(0.1, 0.9, ax=ax0, energy="13 TeV",simulation=True,lumi=139)
    

    return fig


def Plot_ROC(Extraction_Results, output_path, period, reweighting_var, reweighting_option):
    swaped_Extraction_Results = {}
    label_ptrange = [500, 600, 800, 1000, 1200, 1500, 2000]
    label_var = ['jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'GBDT_newScore']
    label_keys = [*Extraction_Results['jet_nTracks'][500]]
    for l_ptrange in label_ptrange[:-1]:
        swaped_Extraction_Results[l_ptrange] = {}
        for l_key in label_keys:
            swaped_Extraction_Results[l_ptrange][l_key] = {}
            for l_var in label_var:
                swaped_Extraction_Results[l_ptrange][l_key][l_var] = Extraction_Results[l_var][l_ptrange][l_key]

    output_path_new = output_path / period / "ROCs" / f"{reweighting_var}_{reweighting_option}" 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok =True)

    eta_regions = {
        "ForwardandCentral": ['p_Quark', 'p_Gluon'],
        "Forward": ['p_Forward_Quark', 'p_Forward_Gluon'],
        "Central": ['p_Central_Quark', 'p_Central_Gluon']
    }
    for k, v in eta_regions.items():
        for l_ptrange in label_ptrange[:-1]:
            two_vars = swaped_Extraction_Results[l_ptrange]
            fig = _Plot_ROC(p_Quark_unumpy = two_vars[v[0]], p_Gluon_unumpy = two_vars[v[1]],
                            l_ptrange=l_ptrange, etaregion=k)
            fig_name = output_path_new / f"ROC_{l_ptrange}_{k}_{reweighting_option}.pdf"
            fig.savefig(fig_name)
            plt.close()

def Plot_Fraction(Extraction_Results, output_path, period, reweighting_var, reweighting_option):
    fraction_pt_slices = []
    for pt in label_pt_bin[:-1]:
        fraction_pt_slices.append(Extraction_Results['jet_pt'][pt]['f'])

    fraction_pt_slices = np.array(fraction_pt_slices)
    fractions = fraction_pt_slices.reshape((6, 4)).swapaxes(0,1)
    frac_Forward_Quark = fractions[0, :]
    frac_Forward_Gluon = fractions[1, :]
    frac_Central_Quark = fractions[2, :]
    frac_Central_Gluon = fractions[3, :]

    output_path_new = output_path / period / "Fractions" / f"{reweighting_var}_{reweighting_option}" 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok =True)

    fig, ax = plt.subplots()
    bin_edges = label_pt_bin
    ax.stairs(frac_Forward_Quark, bin_edges, label=r"$f_{Higher, Quark}$", color="purple", baseline=None, linewidth = 2)
    ax.stairs(frac_Forward_Gluon, bin_edges, label=r"$f_{Higher, Gluon}$", color="red", baseline=None, linewidth = 2)
    ax.stairs(frac_Central_Quark, bin_edges, label=r"$f_{Lower, Quark}$", color="blue", baseline=None, linewidth = 2)
    ax.stairs(frac_Central_Gluon, bin_edges, label=r"$f_{Lower, Gluon}$", color="green", baseline=None, linewidth = 2)
    ax.legend()

    ax.hlines(y=0.5, xmin=bin_edges[0], xmax=bin_edges[-1], linestyles='dashed', color="black")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ampl.plot.set_xlabel('Jet $p_{\mathrm{T}}$ [GeV]')
    ax.set_ylabel('Fraction') 
    ampl.plot.draw_atlas_label(0.1, 0.9, ax=ax, energy="13 TeV", simulation=True,lumi=139)

    fig_name = output_path_new / f"Fraction.pdf"
    fig.savefig(fig_name)



def Plot_ForwardCentral_MCvsData(pt, var, output_path, period, reweighting_var, reweighting_option, 
                                 Forward_MC, Central_MC, Forward_Data, Central_Data, if_norm, show_yields=True):

    bin_edges = HistBins[var]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Forward_MC), yerr=unumpy.std_devs(Forward_MC), color = 'blue', label = 'Higher MC', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Central_MC), yerr=unumpy.std_devs(Central_MC), color = 'red', label = 'Lower MC', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Forward_Data), yerr=unumpy.std_devs(Forward_Data), color = 'blue', label = 'Higher Data', marker='.', linestyle="none")
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(Central_Data), yerr=unumpy.std_devs(Central_Data), color = 'red', label = 'Lower Data', marker='.', linestyle="none")    
    ax0.legend()
    ax0.set_xlim(bin_edges[0], bin_edges[-1])
    ampl.plot.draw_atlas_label(0.1, 0.85, ax=ax0, energy="13 TeV",lumi=139)
    #ax0.set_title(f"{pt} GeV: MC vs Data " + rf"{Map_var_title[var]}"  + f" distribution, {reweighting_option}")
    if show_yields and not if_norm:
        n_Forward_MC = np.sum(unumpy.nominal_values(Forward_MC))
        n_Central_MC = np.sum(unumpy.nominal_values(Central_MC))
        n_Forward_Data = np.sum(unumpy.nominal_values(Forward_Data))
        n_Central_Data = np.sum(unumpy.nominal_values(Central_Data))
        ax0.text(x=0.3, y=0.04, 
                s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                    f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e}",
                   ha='left', va='bottom', transform=ax0.transAxes)
        

    ratio_Forward = safe_array_divide_unumpy(numerator = Forward_Data, denominator = Forward_MC)
    ratio_Central = safe_array_divide_unumpy(numerator = Central_Data, denominator = Central_MC)

    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Forward), yerr=unumpy.std_devs(ratio_Forward), color = 'blue', label = 'Higher', drawstyle='steps-mid')
    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Central), yerr=unumpy.std_devs(ratio_Central), color = 'red', label = 'Lower', drawstyle='steps-mid')
    ax1.set_ylabel("Data/MC")
    ax1.set_ylim(0.7, 1.3)
    ax1.legend()
    #ax1.set_xlabel(f"{Map_var_title[var]}")
    ampl.plot.set_xlabel(f"{Map_var_title[var]}")
    ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--', label='ratio = 1')
    ax1.plot()

    output_path_new = output_path / period / "FvsC" / f"{reweighting_var}_{reweighting_option}" / var 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok =True)
    if if_norm == True:
        ax0.set_ylabel("Normalized")
        fig_name = output_path_new / f"MCvsData_FvsC_{pt}_{var}_{reweighting_option}_normed.pdf"
    else: 
        ax0.set_ylabel("Yields")
        fig_name = output_path_new / f"MCvsData_FvsC_{pt}_{var}_{reweighting_option}.pdf"
    fig.savefig(fig_name)
    plt.close()

def Plot_Parton_ForwardvsCentral(pt, var, output_path, period, reweighting_var, reweighting_option, p_Forward_Quark, p_Central_Quark, p_Forward_Gluon, p_Central_Gluon ):
    bin_edges = HistBins[var]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    # breakpoint()
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(p_Forward_Quark), yerr=unumpy.std_devs(p_Forward_Quark), color = 'blue', label = 'Higher  Quark', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(p_Central_Quark), yerr=unumpy.std_devs(p_Central_Quark), color = 'blue', label = 'Lower Quark', linestyle='--', drawstyle='steps-mid')
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(p_Forward_Gluon), yerr=unumpy.std_devs(p_Forward_Gluon), color = 'red', label = 'Higher Gluon', drawstyle="steps-mid")
    ax0.errorbar(x=bin_centers, y=unumpy.nominal_values(p_Central_Gluon), yerr=unumpy.std_devs(p_Central_Gluon), color = 'red', label = 'Lower Gluon', linestyle='--', drawstyle='steps-mid')    
    ax0.set_xlim(bin_edges[0], bin_edges[-1])
    ax0.set_ylabel("Normalized")
    ax0.legend()
    if var=="ntrk":
        ampl.plot.draw_atlas_label(0.6, 0.6, ax=ax0, energy="13 TeV",simulation=True,lumi=139)
    elif pt==500:
        ampl.plot.draw_atlas_label(0.55, 0.9, ax=ax0, energy="13 TeV",simulation=True,lumi=139)
    else:
        ampl.plot.draw_atlas_label(0.1, 0.85, ax=ax0, energy="13 TeV",simulation=True,lumi=139)
    #ax0.set_title(f"{pt} GeV: MC Q/G " + rf"{Map_var_title[var]}, {reweighting_option}")

    ratio_Quark = safe_array_divide_unumpy(numerator = p_Forward_Quark, denominator = p_Central_Quark)
    ratio_Gluon = safe_array_divide_unumpy(numerator = p_Forward_Gluon, denominator = p_Central_Gluon)

    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Quark), yerr=unumpy.std_devs(ratio_Quark), color = 'blue', label = 'Quark', drawstyle='steps-mid')
    ax1.errorbar(x=bin_centers, y=unumpy.nominal_values(ratio_Gluon), yerr=unumpy.std_devs(ratio_Gluon), color = 'red', label = 'Gluon', drawstyle='steps-mid')

    ax1.set_ylabel("Higher/Lower")
    ax1.set_ylim(0.7, 1.3)
    ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
    ampl.plot.set_xlabel(f"{Map_var_title[var]}")
    ax1.legend()
    ax1.plot()
    output_path_new = output_path / period / "FvsC" / f"{reweighting_var}_{reweighting_option}" / var 
    if not output_path_new.exists():
        output_path_new.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path_new / f"MC_truth_Q_G_FvsC_{pt}_{var}_{reweighting_option}.pdf")
    plt.close()


def Plot_Extracted_unumpy(pt, var, output_path, period, reweighting_var, reweighting_factor, p_Quark, extract_p_Quark, p_Gluon, extract_p_Gluon, extract_p_Quark_Data, extract_p_Gluon_Data,
                          show_yields=False, n_Forward_MC=None, n_Central_MC=None, variances_Forward_MC = None, variances_Central_MC = None,
                          n_Forward_Data=None, n_Central_Data=None, variances_Forward_Data=None, variances_Central_Data=None):
    bin_edges = HistBins[var]
    bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])
        
    jet_types = ["quark", "gluon"]
    color_types = ["blue", "red"]
    plot_data = [[p_Quark, extract_p_Quark, extract_p_Quark_Data], 
                 [p_Gluon, extract_p_Gluon, extract_p_Gluon_Data]]
    plot_data_bin_content = unumpy.nominal_values(plot_data) 
    plot_data_bin_error = unumpy.std_devs(plot_data)

    for i, jet_type in enumerate(jet_types):  # i is the idx of jet type
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})

        # ax0.stairs(values = plot_data[i][0], edges = bin_edges, color = color_types[i], label = f'{jet_type}, extracted MC', baseline=None)
        # ax0.stairs(values = plot_data[i][1], edges = bin_edges, color = color_types[i], linestyle='--', label = f'{jet_type}, truth MC', baseline=None)
        # ax0.stairs(values = plot_data[i][2], edges = bin_edges, color = color_types[i], linestyle=':', label = f'{jet_type}, extracted Data', baseline=None)
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][1], yerr = plot_data_bin_error[i][1], drawstyle = 'steps-mid', label = "Extracted MC", color= "red")
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][2], yerr = plot_data_bin_error[i][2], drawstyle = 'steps-mid', label = "Extracted Data", color= "black", linestyle='', marker= "o")
        
        ax0.set_xlim(bin_edges[0], bin_edges[-1])
        ax0.legend()

        y_max = np.max(plot_data_bin_content)
        ax0.set_ylim(-0.01, y_max * 1.3)
        ax0.set_ylabel("Normalized")
#        ax0.set_title(f"{pt} GeV {jet_type}: Extracted " + rf"{Map_var_title[var]}"  + f" distribution, {reweighting_factor}")
        ampl.plot.draw_atlas_label(0.1, 0.9, ax=ax0, energy="13 TeV",lumi=139)
        if show_yields:
            ax0.text(x=0.3, y=0.04, 
            s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                f"MC forward variances:{variances_Forward_MC:.2e},central variances:{variances_Central_MC:.2e} \n" +
                f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e} \n"+
                f"Data forward variances:{variances_Forward_Data:.2e}, central variances:{variances_Central_Data:.2e}",
            ha='left', va='bottom', transform=ax0.transAxes)

        ratio_data_over_extractedMC = safe_array_divide_unumpy(plot_data[i][2], plot_data[i][1])
        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(ratio_data_over_extractedMC), yerr = unumpy.std_devs(ratio_data_over_extractedMC), drawstyle = 'steps-mid', color= "black", linestyle='', marker= "o")

        #ax1.legend()
        ax1.set_ylim(0.7,1.3)
        ax1.set_ylabel("Data/MC")
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ampl.plot.set_xlabel(f"{Map_var_title[var]}")
        ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
        output_path_new = output_path / period / "Extractions" /f"{reweighting_var}_{reweighting_factor}"  / var 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)

        fig.tight_layout()
        fig.savefig( output_path_new / f"DataExtraction_{pt}_{jet_type}_{var}.pdf")
        plt.close()

def Plot_Closure_unumpy(pt, var, output_path, period, reweighting_var, reweighting_factor, p_Quark, extract_p_Quark, p_Gluon, extract_p_Gluon, extract_p_Quark_Data, extract_p_Gluon_Data,
                          show_yields=False, n_Forward_MC=None, n_Central_MC=None, variances_Forward_MC = None, variances_Central_MC = None,
                          n_Forward_Data=None, n_Central_Data=None, variances_Forward_Data=None, variances_Central_Data=None):
    bin_edges = HistBins[var]
    bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])
        
    jet_types = ["quark", "gluon"]
    color_types = ["blue", "red"]
    plot_data = [[p_Quark, extract_p_Quark, extract_p_Quark_Data], 
                 [p_Gluon, extract_p_Gluon, extract_p_Gluon_Data]]
    plot_data_bin_content = unumpy.nominal_values(plot_data) 
    plot_data_bin_error = unumpy.std_devs(plot_data)

    for i, jet_type in enumerate(jet_types):  # i is the idx of jet type
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})

        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][0], yerr = plot_data_bin_error[i][0], drawstyle = 'steps-mid', label = "Truth MC",color="blue")
        ax0.errorbar(x = bin_centers, y = plot_data_bin_content[i][1], yerr = plot_data_bin_error[i][1], drawstyle = 'steps-mid', label = "Extracted MC",color="red")
        
        ax0.set_xlim(bin_edges[0], bin_edges[-1])
        ax0.legend()

        y_max = np.max(plot_data_bin_content)
        ax0.set_ylim(-0.01, y_max * 1.3)
        ax0.set_ylabel("Normalized")
#        ax0.set_title(f"{pt} GeV {jet_type}: MCClosure " + rf"{Map_var_title[var]}"  + f" distribution, {reweighting_factor}")
        ampl.plot.draw_atlas_label(0.1, 0.9, ax=ax0, energy="13 TeV",lumi=139,simulation=True)
        if show_yields:
            ax0.text(x=0.3, y=0.04, 
            s = f"MC forward yield:{n_Forward_MC:.2e},central yield:{n_Central_MC:.2e} \n"+
                f"MC forward variances:{variances_Forward_MC:.2e},central variances:{variances_Central_MC:.2e} \n" +
                f"Data forward yield:{n_Forward_Data:.2e}, central yield:{n_Central_Data:.2e} \n"+
                f"Data forward variances:{variances_Forward_Data:.2e}, central variances:{variances_Central_Data:.2e}",
            ha='left', va='bottom', transform=ax0.transAxes)

        ratio_truthMC_over_extractedMC = safe_array_divide_unumpy(plot_data[i][0], plot_data[i][1])
        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(ratio_truthMC_over_extractedMC), yerr = unumpy.std_devs(ratio_truthMC_over_extractedMC), drawstyle = 'steps-mid',color="red")

        #ax1.legend()
        ax1.set_ylim(0.7,1.3)
        ax1.set_ylabel("Truth/Extracted")
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ampl.plot.set_xlabel(f"{Map_var_title[var]}")
        ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
        output_path_new = output_path / period / "MCClosure" /f"{reweighting_var}_{reweighting_factor}"  / var 
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)

        fig.tight_layout()
        fig.savefig( output_path_new / f"MCClosure_{pt}_{jet_type}_{var}.pdf")
        plt.close()

def Plot_WP(WP, var, output_path, period, reweighting_var, reweighting_factor,
            quark_effs, gluon_rejs, quark_effs_data, gluon_rejs_data,
            if_save=True):
    
    SF_quark = safe_array_divide_unumpy(quark_effs_data, quark_effs)
    SF_gluon = safe_array_divide_unumpy(gluon_rejs_data, gluon_rejs)

    if if_save:
        bin_edges = np.array([500, 600, 800, 1000, 1200, 1500, 2000])
        bin_centers = 1/2 * (bin_edges[:-1] + bin_edges[1:])

        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
        ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs), yerr = unumpy.std_devs(quark_effs), label = "Quark Efficiency, Extracted MC", color = "blue",linestyle='none', marker='^')
        ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs), yerr = unumpy.std_devs(gluon_rejs), label = "Gluon Rejection, Extracted MC", color = "red",linestyle='none', marker='^')
        ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(quark_effs_data), yerr = unumpy.std_devs(quark_effs_data), label = "Quark Efficiency, Extracted Data", color= "blue", linestyle='none', marker= "o")
        ax0.errorbar(x = bin_centers, y = unumpy.nominal_values(gluon_rejs_data), yerr = unumpy.std_devs(gluon_rejs_data), label = "Gluon Rejection, Extracted Data",color= "red", linestyle='none', marker= "o")
        ax0.legend()
        ax0.set_yticks(np.linspace(0, 1, 21))
        ax0.set_xticks(bin_edges)
        ax0.set_ylim(0.4, 1.2)
        ax0.set_xlim(bin_edges[0], bin_edges[-1])
        ax0.set_ylabel("Efficiency or Rejection")

        ax0.grid()
#        ax0.set_title(f"{var} for extracted q/g at {WP} WP")
        ampl.plot.draw_atlas_label(0.1, 0.9, ax=ax0, energy="13 TeV",lumi=139)


        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_quark), yerr = unumpy.std_devs(SF_quark), linestyle='none', label = "quark SF", marker='.')
        ax1.errorbar(x = bin_centers, y = unumpy.nominal_values(SF_gluon), yerr = unumpy.std_devs(SF_gluon), linestyle='none', label = "gluon SF", marker='.')
        ax1.legend(fontsize = 'x-small')
        ax1.set_ylim(0.7, 1.3)
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ampl.plot.set_xlabel(f"{Map_var_title[var]}")
        ax1.set_xticks(bin_edges)
        ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'gray', linestyle = '--')
        ax1.set_ylabel("SFs")

        output_path_new = output_path / period / "WPs" / f"{reweighting_var}_{reweighting_factor}" / var
        if not output_path_new.exists():
            output_path_new.mkdir(parents = True)
        fig.savefig( output_path_new/ f"{reweighting_var}_WP_{WP}.pdf")
        plt.close()

    return SF_quark, SF_gluon

def WritePickle(obj, name, output_path, period, reweighting_var, reweighting_factor):
    output_path_new = output_path / period / f"{name}_pkls" / f"{reweighting_var}_{reweighting_factor}"

    if not output_path_new.exists():
        output_path_new.mkdir(parents = True)
    
    pkl_file_name = output_path_new / f"{name}.pkl"
    logging.info(f"Writing {name} to the pickle file: {pkl_file_name}")
    joblib.dump(obj, pkl_file_name)

def WriteSFtoPickle(Hist_SFs, output_path, period, reweighting_var, reweighting_factor):
    WritePickle(Hist_SFs, "SFs", output_path, period, reweighting_var, reweighting_factor)
    
def WriteWPcuttoPickle(WP_cuts, output_path, period, reweighting_var, reweighting_factor):
    WritePickle(WP_cuts, "WP_cuts", output_path, period, reweighting_var, reweighting_factor)


def Calculate_SF(input_mc_path, input_data_path, period, reweighting_factor, output_path):
    # label_var = ['ntrk', 'bdt']
    
    reweighting_map = {
        "none" : "NoReweighting",
        "quark" : "Reweighting_Quark_Factor",
        "gluon" : "Reweighting_Gluon_Factor"
    }

    for reweighting_var in ["ntrk", "newBDT"]:
        # HistMap_MC, HistMap_Error_MC, HistMap_MC_unumpy = Read_Histogram_Root(input_mc_path, sampletype="MC", code_version="new", reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        # HistMap_Data, HistMap_Error_Data, HistMap_Data_unumpy = Read_Histogram_Root(input_data_path, sampletype="Data", code_version="new", reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        HistMap_MC_unumpy = Read_Histogram_pkl(input_mc_path, reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        HistMap_Data_unumpy = Read_Histogram_pkl(input_data_path, reweighting_var=reweighting_var, reweighting_factor=reweighting_factor)
        # breakpoint()
        #### Draw pt spectrum
        Plot_Pt_Spectrum(HistMap_MC_unumpy, HistMap_Data_unumpy, output_path, reweighting_var, reweighting_map[reweighting_factor])


        Extraction_Results = Extract(HistMap_MC_unumpy, HistMap_Data_unumpy)
        joblib.dump(Extraction_Results, output_path / f"{reweighting_var}_Extraction_Results.pkl")
        #### Draw ROC plot 
        Plot_ROC(Extraction_Results, output_path, period, reweighting_var, reweighting_option=reweighting_map[reweighting_factor])

        WPs = [0.5, 0.6, 0.7, 0.8]
        SFs = {}

        for var in label_var:
            SFs[var] = {}
            for l_pt in label_ptrange[:-1]:
                Extraction_var_pt =  Extraction_Results[var][l_pt]
                #### Draw Forward vs Central plots 
                Plot_ForwardCentral_MCvsData(pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_option= reweighting_map[reweighting_factor],
                                    Forward_MC= Extraction_var_pt['Forward_MC'], 
                                    Central_MC= Extraction_var_pt['Central_MC'],
                                    Forward_Data= Extraction_var_pt['Forward_Data'], 
                                    Central_Data= Extraction_var_pt['Central_Data'],
                                    if_norm=False, show_yields=False)

                Plot_ForwardCentral_MCvsData(pt = l_pt, var = var, output_path = output_path, 
                                    period = period, reweighting_var = reweighting_var,
                                    reweighting_option= reweighting_map[reweighting_factor],
                                    Forward_MC= Normalize_unumpy(Extraction_var_pt['Forward_MC']), 
                                    Central_MC= Normalize_unumpy(Extraction_var_pt['Central_MC']),
                                    Forward_Data= Normalize_unumpy(Extraction_var_pt['Forward_Data']), 
                                    Central_Data= Normalize_unumpy(Extraction_var_pt['Central_Data']),
                                    if_norm=True, show_yields=False)

                Plot_Parton_ForwardvsCentral(pt = l_pt, var = var, output_path = output_path,
                                    period = period, reweighting_var = reweighting_var,
                                    reweighting_option = reweighting_map[reweighting_factor], 
                                    p_Forward_Quark = Normalize_unumpy(Extraction_var_pt['Forward_Quark']),  
                                    p_Central_Quark = Normalize_unumpy(Extraction_var_pt['Central_Quark']), 
                                    p_Forward_Gluon = Normalize_unumpy(Extraction_var_pt['Forward_Gluon']), 
                                    p_Central_Gluon = Normalize_unumpy(Extraction_var_pt['Central_Gluon'])
                                    )

                #### Draw extraction plots 
                Plot_Extracted_unumpy(pt = l_pt, var= var, output_path= output_path, 
                                        period= period, reweighting_var = reweighting_var,
                                        reweighting_factor= reweighting_map[reweighting_factor],
                                        p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                        extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                        extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                        show_yields=False, 
                                        n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                        n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                        variances_Forward_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Forward_MC'])**2),
                                        variances_Central_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Central_MC'])**2), 
                                        n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                        n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])),
                                        variances_Forward_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Forward_Data'])**2),
                                        variances_Central_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Central_Data'])**2)
                                        )
                Plot_Closure_unumpy(pt = l_pt, var= var, output_path= output_path, 
                                        period= period, reweighting_var = reweighting_var,
                                        reweighting_factor= reweighting_map[reweighting_factor],
                                        p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                        extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                        extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                        show_yields=False, 
                                        n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                        n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                        variances_Forward_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Forward_MC'])**2),
                                        variances_Central_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Central_MC'])**2), 
                                        n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                        n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])),
                                        variances_Forward_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Forward_Data'])**2),
                                        variances_Central_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Central_Data'])**2)
                                        )
            #### Draw working points 
            for WP in WPs:
                SFs[var][WP] = {}
                quark_effs_at_pt = []
                gluon_rejs_at_pt = []
                quark_effs_data_at_pt = []
                gluon_rejs_data_at_pt = []
                for ii, l_pt in enumerate(label_ptrange[:-1]):
                    extract_p_Quark_MC =  Extraction_Results[var][l_pt]['extract_p_Quark_MC']
                    extract_p_Gluon_MC =  Extraction_Results[var][l_pt]['extract_p_Gluon_MC']
                    extract_p_Quark_Data =  Extraction_Results[var][l_pt]['extract_p_Quark_Data']
                    extract_p_Gluon_Data =  Extraction_Results[var][l_pt]['extract_p_Gluon_Data']

                    extract_p_Quark_cum_sum = np.cumsum(unumpy.nominal_values(extract_p_Quark_MC))
                    cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0]+1
                    

                    quark_effs_at_pt.append(np.sum(extract_p_Quark_MC[:cut])) 
                    gluon_rejs_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))
                    quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                    gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))

                SF_quark, SF_gluon = Plot_WP(WP = WP, var= var, output_path= output_path, 
                        period= period, reweighting_var = reweighting_var,
                        reweighting_factor= reweighting_map[reweighting_factor],
                        quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                        quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt)
                SFs[var][WP]["Quark"] = SF_quark
                SFs[var][WP]["Gluon"] = SF_gluon

            #WriteSFtoPickle(var = var,Hist_SFs = SFs, output_path=output_path, period=period, reweighting_var = reweighting_var,
            #            reweighting_factor= reweighting_map[reweighting_factor])
        if reweighting_factor == "none":
            break 

if __name__ == '__main__':
    """This script do the matrix method and calculate SFs and save them to pickle files. 
       It generates the following structure for a successful run. 
       <output-path>
           └── <period>
                ├── Extractions
                ├── FvsC
                ├── Pt_spectrum
                ├── ROCs
                ├── SFs_pkls
                └── WPs

    Raises:
        Exception: if the mc_file is not a root file, raise an error. 
        Exception: if the input file is not consistent with the period passed, raise an error. 
    """
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
    parser.add_argument('--path-mc', help='The path to the MC histogram file(.root file).')
    parser.add_argument('--path-data', help='The path to the Data histogram file(.root file).')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"])
    parser.add_argument('--reweighting', help='The reweighting method', choices=['none', 'quark', 'gluon'])
    parser.add_argument('--output-path', help='Output path')
    args = parser.parse_args()

    mc_file_path = Path(args.path_mc)
    data_file_path = Path(args.path_data)
    output_path = Path(args.output_path)
    period = args.period

    if not output_path.exists():
        output_path.mkdir(parents=True)

    Calculate_SF(input_mc_path=mc_file_path, input_data_path=data_file_path, period = period, reweighting_factor = args.reweighting , output_path = output_path)
    
