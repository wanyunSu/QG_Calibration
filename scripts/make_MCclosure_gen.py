#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib 
import numpy as np 
from matplotlib import pyplot as plt 
from uncertainties import ufloat, unumpy

from pathlib import Path
from tqdm import tqdm
from numpy import array
import sys
sys.path.append("..")
from core.utils import *
import atlas_mpl_style as ampl
import random
import mplhep as hep
hep.style.use("ATLAS")
import random

#pkl_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots'
#pkl_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots_prob'  # probability
pkl_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots_new'
pkl_path = Path(pkl_path)
nominal_path = pkl_path / 'nominal' / 'plots' / 'ADE' / 'Extraction_Results'


# In[2]:


reweighting_vars = ['none','jet_nTracks', 'GBDT_newScore']
nominal_keys = [reweighting_var + '_quark_reweighting_weights' for reweighting_var in reweighting_vars]
WPs = [0.5]#, 0.6, 0.7, 0.8]
partons=['Quark','Gluon']
label_var = ['jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'GBDT_newScore']
label_ptrange = [500, 600, 800, 1000, 1200, 1500,2000]
label_eta_bin = [0, 0.5, 1.0, 1.5, 2.1]
bin_centers = 0.5 * (np.array(label_ptrange[:-1]) + np.array(label_ptrange[1:]))
Map_var_title = {
    "jet_pt": "$p_{T}$",
    "jet_nTracks": "$n_{trk}$",
    "jet_trackBDT": "old BDT",
    "jet_eta": "$\eta$",
    "jet_trackC1": "$C_{1}$",
    "jet_trackWidth": "$w^{trk}$",
    "GBDT_newScore": "BDT score"
}
HistBins = {
    'jet_pt' : np.linspace(500, 2000, 61),
    'jet_eta' : np.linspace(-2.5, 2.5, 51), 
    'jet_nTracks' : np.linspace(-0.5, 59.5, 61),
    'jet_trackWidth' : np.linspace(0, 0.4, 61),
    'jet_trackC1' : np.linspace(0, 0.4, 61),
    'jet_trackBDT' : np.linspace(-1.0, 1.0, 101),
    'GBDT_newScore' : np.linspace(-5.0, 5.0, 101),
}
Map_rvar={
   "none": "(No re-weight)",
   'jet_nTracks' : "(Re-weighted)",
   'GBDT_newScore' : "(Re-weighted)",

}

Map_gen={
"nominal": "PYTHIA",
"hadronization/sherpa": "SHERPA (Cluster Had.)",
"hadronization/sherpalund": "SHERPA (String Had.)",
"matrix_element/powhegpythia":"POWHEG + PYTHIA",
"parton_shower/herwigangle":"HERWIG (Angular Shower)",
"parton_shower/herwigdipole":"HERWIG (Dipole Shower)",
}


# In[3]:


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

markers=['o','s','v','d','p','x','*','^']
ind = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
ind_all = ind(15)





# In[ ]:


SFs = {}
Extraction_Results={}
syst_list=['nominal','matrix_element/powhegpythia','parton_shower/herwigangle','parton_shower/herwigdipole','hadronization/sherpa','hadronization/sherpalund']
label_var = ['jet_nTracks', 'jet_trackWidth', 'jet_trackC1', 'GBDT_newScore']

for i_pt, pt in enumerate(label_ptrange[:-1]):
    for rvar in reweighting_vars:
        for var in label_var:
            bin_edges = HistBins[var]
            bin_width = bin_edges[1]-bin_edges[0]
            #bin_centers1 = 1/2 * (bin_edges[:-1] + bin_edges[1:])
            for parton in partons:
                plt.rcParams['figure.figsize'] = [6, 6]

                #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.1})
                fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
                for i, syst in enumerate(syst_list):
                    syst_path= pkl_path / syst / 'plots' / 'ADE' / 'Extraction_Results'
                    Extraction_Results[syst]= joblib.load(syst_path / f'{rvar}_Extraction_Results.pkl' )
                
                    Extraction_var_pt =  Extraction_Results[syst][var][pt]
        
                    syst_truth = Extraction_Results[syst][var][pt][f'p_{parton}']*bin_width       
                    syst_Q = Extraction_Results[syst][var][pt][f'extract_p_{parton}_MC']*bin_width
                    syst_data=Extraction_Results[syst][var][pt][f'extract_p_{parton}_Data']*bin_width
                    nom_data=Extraction_Results['nominal'][var][pt][f'extract_p_{parton}_Data']*bin_width

                    y_max = np.max(unumpy.nominal_values(syst_truth))


                    #hep.histplot(unumpy.nominal_values(syst_truth),bins=bin_edges,label=f'Truth {syst}',ax=ax0,histtype='step',linestyle='--')
                    #hep.histplot(unumpy.nominal_values(syst_Q),bins=bin_edges,label=f'Extracted {syst}',ax=ax0,histtype='step',linestyle='-')
                    hep.histplot(unumpy.nominal_values(syst_Q),bins=bin_edges,label=Map_gen[syst],ax=ax0,histtype='errorbar',marker=markers[i],yerr=unumpy.std_devs(syst_Q),markersize=7,mfc='none')#,color=ind_all[i])


                
                    ytext=y_max * 2
                    ax0.set_ylim(0, ytext *1.2)

                    ax0.set_ylabel("Fraction of events",fontsize=15)

                    ratio_truthMC_over_extractedMC = safe_array_divide_unumpy(syst_Q,nom_data)

                    #ratio_truthMC_over_extractedMC = safe_array_divide_unumpy(syst_Q,syst_truth)
                    hep.histplot(unumpy.nominal_values(ratio_truthMC_over_extractedMC),bins=bin_edges,ax=ax1,histtype='errorbar',marker=markers[i],yerr=unumpy.std_devs(ratio_truthMC_over_extractedMC),markersize=7,mfc='none')
                    #hep.histplot(unumpy.nominal_values(ratio_truthMC_over_extractedMC),bins=bin_edges,ax=ax1,histtype='errorbar',marker=".",color='black',yerr=unumpy.std_devs(ratio_truthMC_over_extractedMC),xerr=True,markersize=4)
                hep.histplot(unumpy.nominal_values(nom_data),bins=bin_edges,label='Data',ax=ax0,histtype='errorbar',marker="o",color='black',yerr=unumpy.std_devs(nom_data),markersize=6)

                ax1.set_ylim(0.6,1.4)
                    #ax2.set_ylim(0.6,1.4)
                #ax1.set_ylabel("MC closure",loc='center',fontsize=15)
                ax1.set_ylabel("Ratio to Data",loc='center',fontsize=15)
                    #ax1.set_ylabel(r"$\frac{Extracted~MC}{Truth~MC}$",loc='center')
                    #ax2.set_ylabel(r"$\frac{Data}{MC}$",loc='center')
        
                hep.atlas.label(ax=ax0,data=True,lumi=140,fontsize=12)
                ax0.legend(loc='upper right',fontsize=10)

                ax0.text(bin_edges[3],ytext*0.94,f'{pt} GeV < '+ r'$p_{\mathrm{T}}$'+ f' < {label_ptrange[i_pt+1]} GeV ',fontsize=12)                
                ax0.text(bin_edges[3],ytext*0.84,f'{parton}-jets',fontsize=12)                
                ax1.set_xlim(bin_edges[0], bin_edges[-1])
                ax1.set_xlabel(f"{Map_var_title[var]}")
                ax1.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
#                y_ticklabels = [f'{val:.{2}f}' for val in ax0.get_yticks()]
#                ax0.set_yticklabels(y_ticklabels)
                #ax2.hlines(y = 1, xmin = bin_edges[0], xmax = bin_edges[-1], color = 'black', linestyle = '--')
                fig.align_labels()

                fig.savefig(f'../mcclosure_gen/mcclosure_{rvar}_{var}_{parton}_{pt}_data.pdf')
                    #fig.savefig(f'./mcclosure_sf/mcclosure_{var}_{parton}_{pt}_SF.pdf')



# In[4]:


