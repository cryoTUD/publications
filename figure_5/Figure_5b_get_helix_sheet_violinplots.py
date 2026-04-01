#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:18:11 2022

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
import seaborn as sns
import numpy as np
import pandas as pd

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
    ax.set_xlabel('$1/d [\AA^{-1}]$',fontsize=font)
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


local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

with open(os.path.join(input_folder, "secondary_structure_analysis","pwlf_breakpoint_data.pickle"), "rb") as f:
    pwlf_analysis_read = pickle.load(f)

frequency = frequency_array(profile_size=256, apix=0.5)

breakpoints_helix = np.array(pwlf_analysis_read['helix']['breakpoints'])
breakpoints_sheet = np.array(pwlf_analysis_read['sheet']['breakpoints'])

secondary_structure_breakpoint_helix_raw = breakpoints_helix[:,1]
secondary_structure_breakpoint_sheet_raw = breakpoints_sheet[:,1]

low_freq_cutoff = 6
high_freq_cutoff = 3

clean_indices_helix = np.where(np.logical_and(secondary_structure_breakpoint_helix_raw<low_freq_cutoff,secondary_structure_breakpoint_helix_raw>high_freq_cutoff))
clean_indices_sheet = np.where(np.logical_and(secondary_structure_breakpoint_sheet_raw<low_freq_cutoff,secondary_structure_breakpoint_sheet_raw>high_freq_cutoff))

ss_breakpoint_helix = secondary_structure_breakpoint_helix_raw[clean_indices_helix]
ss_breakpoint_sheet = secondary_structure_breakpoint_sheet_raw[clean_indices_sheet]

#%%
from locscale.utils.plot_utils import pretty_violinplots
filename = os.path.join(output_folder, "Figure_5b_Violinplot_helix_sheet_11.eps")
pretty_violinplots(list_of_series=[ss_breakpoint_helix,ss_breakpoint_sheet], xticks=["Helix Profiles","Sheet Profiles"], ylabel="Resolution ($\AA$)", 
                   filename=filename, fontscale=0.5, figsize=(1.5,1), linewidth=3)

    


