#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:26:29 2021

@author: alok
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from locscale.include.emmer.ndimage.profile_tools import frequency_array, crop_profile_between_frequency, estimate_bfactor_standard
from locscale.include.emmer.pdb.pdb_tools import find_wilson_cutoff
import os
import pwlf

def clean_profile_data(input_dictionary):
    clean_dictionary = {}
    for key_1 in input_dictionary.keys():
        temp_dictionary = {}
        clean = True
       # print(key_1)
        for key_2 in input_dictionary[key_1].keys():
      #      print(key_2)
            arr = input_dictionary[key_1][key_2]
            if not isinstance(arr, np.ndarray):
                continue
           # print(arr)
            else:
                array_is_not_finite = not(np.isfinite(arr).any())
                array_is_nan = np.isnan(arr).any()
                
                
                if array_is_nan or array_is_not_finite:
                    clean = False
                
        if clean:
            
            
            clean_dictionary[key_1] = input_dictionary[key_1]
    
    return clean_dictionary

def normalise(x):
    return x/x.max()

def verify_profile(profile):
    '''
    Processing takes place by: taking only wilson region of the profile. Ignore the amplitudes upto 10A

    Parameters
    ----------
    freq : TYPE
        DESCRIPTION.
    profile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    freq, amplitude = profile
    if np.isnan(amplitude).any():
        
        return False
    
    if np.any(amplitude<=0):
        
        return False
    
    wilson_cutoff = find_wilson_cutoff(num_atoms=amplitude[0])

    bfactor = estimate_bfactor_standard(freq, amplitude=amplitude, wilson_cutoff=wilson_cutoff, fsc_cutoff=1, standard_notation=True)
    
    if bfactor > 10:
        return False
    
    else:
        return True
    
#%%
input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/Input"
output_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/Output"


radial_profiles_file = os.path.join(input_folder,'secondary_structure_profiles.pickle')
radial_profiles_file_2 = os.path.join(input_folder,'nucleotide_profiles.pickle')
selected_pdb_pickle_file = os.path.join(input_folder, "selected_pdb.pickle")

with open(radial_profiles_file,'rb') as f:
    secondary_structure_profile_raw = pickle.load(f)
    
secondary_structure_profile_good = clean_profile_data(secondary_structure_profile_raw)
    
with open(radial_profiles_file_2,'rb') as f:
    nucleotide_profile_raw = pickle.load(f)    

with open(selected_pdb_pickle_file, "rb") as f:
    selected_pdb_id = pickle.load(f)
    
nucleotide_profile_good = clean_profile_data(nucleotide_profile_raw)

print("Cleaned input data!")



helix_profiles = []
sheet_profiles = []
rna_profiles = []
dna_profiles = []

for pdbid in secondary_structure_profile_good.keys():
    if pdbid in selected_pdb_id:
        helix_pr = secondary_structure_profile_good[pdbid]['helix']
        freq = frequency_array(helix_pr, apix=0.5)
        if verify_profile((freq,helix_pr)):
            helix_profiles.append(helix_pr)
    
        sheet_pr = secondary_structure_profile_good[pdbid]['sheet']
        if verify_profile((freq,sheet_pr)):
            sheet_profiles.append(sheet_pr)

for index in nucleotide_profile_good.keys():
    rna_id =  nucleotide_profile_good[index]['rna_id']
    dna_id =  nucleotide_profile_good[index]['dna_id']
    
    if rna_id in selected_pdb_id and dna_id in selected_pdb_id:
        rna_pr = nucleotide_profile_good[index]['rna']
        dna_pr = nucleotide_profile_good[index]['dna']
        
        if verify_profile((freq,rna_pr)):
            rna_profiles.append(rna_pr)
        
        if verify_profile((freq,dna_pr)):
            dna_profiles.append(dna_pr)
#%%





#%%

theoretical_profiles = {
    'helix': helix_profiles,
    'sheet': sheet_profiles,
    'rna': rna_profiles,
    'dna': dna_profiles,}


def get_pwlf_fit(freq, amplitudes, wilson_cutoff, fsc_cutoff):
    
    crop_freq, crop_amplitude = crop_profile_between_frequency(freq, amplitudes, wilson_cutoff, fsc_cutoff)
    
    xdata = crop_freq**2
    ydata = np.log(crop_amplitude)
    
    piecewise_linfit = pwlf.PiecewiseLinFit(xdata, ydata)    
    return piecewise_linfit

def pwlf_prediction_breakpoints(piecewise_linfit, num_segments):
    z = piecewise_linfit.fit(n_segments=num_segments)
    breakpoints = np.sqrt(1/z)
    return breakpoints

def pwlf_predict(pwlffit, freq):
    ydata = pwlffit.predict(freq**2)
    amplitudes = np.exp(ydata)
    
    return amplitudes


pwlf_analysis = {}

theoretical_fit_profiles = {}
breakpoints_profiles = {}
prediction_accuracy = {}

for key in theoretical_profiles.keys():
    profile_list = []
    breakpoints_list = []
    r_squared_list = []
    for profile in tqdm(list(theoretical_profiles[key]), desc="Analysis {} profile".format(key)):
        pwlf_fit = get_pwlf_fit(freq, profile, wilson_cutoff=10, fsc_cutoff=1)
        breakpoints = pwlf_prediction_breakpoints(pwlf_fit, num_segments=4)
        r_squared_pwlf_fit = pwlf_fit.r_squared()
        prediction = pwlf_predict(pwlf_fit, freq)
        
        profile_list.append(prediction)
        breakpoints_list.append(breakpoints)
        r_squared_list.append(r_squared_pwlf_fit)
    
    pwlf_analysis[key] = {
        'pwlf_profiles':profile_list,
        'breakpoints':breakpoints_list,
        'qfit':r_squared_list}

#%%%

    
import pickle

with open(os.path.join(output_folder, "pwlf_analysis.pickle"), "wb") as f:
    pickle.dump(pwlf_analysis, f)
    
#%%
def plot_radial_profile_seaborn(freq, list_of_profiles, font=16, ylims=None, crop_first=10, crop_end=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    freq = freq[crop_first:-crop_end]
    
    sns.set_theme(context="paper", font="Helvetica", font_scale=1.5)
    sns.set_style("white")
    kwargs = dict(linewidth=3)

    profile_list = np.array(list_of_profiles)
    average_profile = np.einsum("ij->j", profile_list) / len(profile_list)

    variation = []
    for col_index in range(profile_list.shape[1]):
        col_extract = profile_list[:,col_index]
        variation.append(col_extract.std())

    variation = np.array(variation)
        
    y_max = average_profile + variation
    y_min = average_profile - variation
    
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=freq, y=average_profile[crop_first:-crop_end], **kwargs)
    ax.fill_between(freq, y_min[crop_first:-crop_end], y_max[crop_first:-crop_end], alpha=0.3)
    ax.set_xlabel('Spatial Frequency $1/d [\AA^{-1}]$',fontsize=font)
    ax.set_ylabel('$\mid F \mid $',fontsize=font)
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels([round(1/x,1) for x in ax.get_xticks()])
    ax2.set_xlabel('$d [\AA]$',fontsize=font)
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()
    plt.show()
    
    return fig

    
    
        
#%%   
    
    
    



