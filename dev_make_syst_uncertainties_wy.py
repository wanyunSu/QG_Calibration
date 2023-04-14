#!/usr/bin/env python
# coding: utf-8

# In[21]:


import joblib 
import numpy as np 
from matplotlib import pyplot as plt 
from uncertainties import ufloat, unumpy

from pathlib import Path
from tqdm import tqdm
from numpy import array
from core.utils import *
import atlas_mpl_style as ampl
ampl.use_atlas_style(usetex=False)


# In[22]:


pkl_path = '/global/cfs/projectdirs/atlas/wys/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new_hrzhao'
pkl_path = Path(pkl_path)


# In[23]:


syst_total = joblib.load("/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/syst_uncertainties/syst_total.pkl")
reweighting_vars = ['jet_nTracks', 'GBDT_newScore']
nominal_keys = [reweighting_var + '_quark_reweighting_weights' for reweighting_var in reweighting_vars]
WPs = [0.5, 0.6, 0.7, 0.8]

# In[24]:


from numpy import array
import random

ind = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
ind_all = ind(15)


#reweighting_vars
#reweighting_vars=['jet_nTracks','GBDT_newScore']
label_pt_bin[:-1]


# In[59]:


bin_centers = 0.5 * (np.array(label_pt_bin[:-1]) + np.array(label_pt_bin[1:]))
bin_centers1 = [500,700,900,1100,1350,2000]

for i_var, reweighting_var in enumerate(reweighting_vars):

    nominal_path = pkl_path / 'nominal' / 'plots' / 'ADE' / 'SFs_pkls'
    nominal_SFs = joblib.load(nominal_path / nominal_keys[i_var] / "SFs.pkl") # this includes many vars with WPs 
    
    for WP in WPs:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,10), sharey=False, gridspec_kw={ 'wspace': 0.45})
        quark_nominal_values = unumpy.nominal_values(nominal_SFs[reweighting_var][WP]['Quark'])
        gluon_nominal_values = unumpy.nominal_values(nominal_SFs[reweighting_var][WP]['Gluon'])
        ax[0].scatter(bin_centers, quark_nominal_values, label = "Quark SF",color="blue")
        ax[1].scatter(bin_centers, gluon_nominal_values, label = "Gluon SF",color="red")

        for key in syst_total[reweighting_var].keys():
            indx = list(syst_total[reweighting_var]).index(key)
            indc = ind_all[indx]
            quark_uncertainty = syst_total[reweighting_var][key][WP]['Quark']
            gluon_uncertainty = syst_total[reweighting_var][key][WP]['Gluon']
            if key == 'JESJER':
                key = 'JES/JER'
            ax[0].fill_between(bin_centers1, quark_nominal_values-quark_uncertainty, quark_nominal_values+quark_uncertainty,                             # color='b',
                            alpha=1, label=f'{key}',facecolor='none',edgecolor=indc)

            ax[1].fill_between(bin_centers1, gluon_nominal_values-gluon_uncertainty, gluon_nominal_values+gluon_uncertainty,                             # color='r', 
                            alpha=1, label=f'{key}',facecolor='none',edgecolor=indc)
        # ax.scatter(bin_centers, unumpy.nominal_values(nominal_SFs['jet_nTracks'][WP]['Gluon']), label = "Gluon")
        for i in range(len(ax)):
            ax[i].set_xlim(label_pt_bin[0], label_pt_bin[-1])
            ax[i].set_ylim(0.4, 1.45)
#            ax[i].set_title(f"WP:{WP}, {reweighting_var}")
            #ampl.plot.set_xlabel("Jet pT [GeV]")
            ax[i].set_xlabel("Jet pT [GeV]")
            ax[i].legend(bbox_to_anchor=(1.0, 1.0))
            ampl.draw_atlas_label(0.1, 0.85, ax=ax[i], energy="13 TeV")

        fig.savefig(f"./syst_uncertainties_note/{reweighting_var}_WP{WP}.pdf")


# ## Draw combined syst

# In[31]:



#bin_centers = 0.5 * (np.array(label_pt_bin[:-1]) + np.array(label_pt_bin[1:]))
parton_total_uncertainty = dict.fromkeys(reweighting_vars)
#
for i_var, reweighting_var in enumerate(reweighting_vars):

    nominal_path = pkl_path / 'nominal' / 'plots' / 'ADE' / 'SFs_pkls'
    nominal_SFs = joblib.load(nominal_path / nominal_keys[i_var] / "SFs.pkl") # this includes many vars with WPs 
    parton_total_uncertainty[reweighting_var] = dict.fromkeys(WPs)
    for WP in WPs:
        parton_total_uncertainty[reweighting_var][WP] = dict.fromkeys(['Quark', 'Gluon'])
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10), sharey=True, gridspec_kw={ 'wspace': 0})
        quark_nominal_values = unumpy.nominal_values(nominal_SFs[reweighting_var][WP]['Quark'])
        gluon_nominal_values = unumpy.nominal_values(nominal_SFs[reweighting_var][WP]['Gluon'])
        ax[0].scatter(bin_centers, quark_nominal_values, label = "Quark SF",color="blue")
        ax[1].scatter(bin_centers, gluon_nominal_values, label = "Gluon SF",color="red")

        quark_total_uncertainty = []
        gluon_total_uncertainty = []

        for key in syst_total[reweighting_var].keys():
            quark_total_uncertainty.append(syst_total[reweighting_var][key][WP]['Quark'])
            gluon_total_uncertainty.append(syst_total[reweighting_var][key][WP]['Gluon'])

        quark_total_uncertainty = np.sqrt(np.sum(np.power(quark_total_uncertainty, 2), axis=0))
        gluon_total_uncertainty = np.sqrt(np.sum(np.power(gluon_total_uncertainty, 2), axis=0))
        
        ax[0].fill_between(bin_centers1, quark_nominal_values-quark_total_uncertainty, quark_nominal_values+quark_total_uncertainty,                             facecolor='blue', alpha=0.2, label=f'Quark total uncertainty')

        ax[1].fill_between(bin_centers1, gluon_nominal_values-gluon_total_uncertainty, gluon_nominal_values+gluon_total_uncertainty,                             facecolor='red', alpha=0.2, label=f'Gluon total uncertainty')
        parton_total_uncertainty[reweighting_var][WP]['Quark'] = quark_total_uncertainty
        parton_total_uncertainty[reweighting_var][WP]['Gluon'] = gluon_total_uncertainty
        # ax.scatter(bin_centers, unumpy.nominal_values(nominal_SFs['jet_nTracks'][WP]['Gluon']), label = "Gluon")
        for i in range(len(ax)):
            ax[i].set_xlim(label_pt_bin[0], label_pt_bin[-1])
            ax[i].set_ylim(0.4, 1.4)
            #ax[i].set_title(f"WP:{WP}, {reweighting_var}")
            ax[i].set_xlabel("Jet pT [GeV]")
            ax[i].legend()
            ampl.draw_atlas_label(0.1, 0.85, ax=ax[i], energy="13 TeV")

        fig.savefig(f"./syst_uncertainties_note/{reweighting_var}_WP{WP}_combined.pdf")
#
#
## In[18]:
#
#
#joblib.dump(syst_total, "syst_total.pkl")
#
#
## In[19]:
#
#
#joblib.dump(parton_total_uncertainty, "parton_total_uncertainty.pkl")
#
#
## In[23]:
#
#
#syst_total['jet_nTracks']['splitting_kernel']
#
#
## In[20]:
#
#
#parton_total_uncertainty
#
#
## In[ ]:
#
#
#
#
