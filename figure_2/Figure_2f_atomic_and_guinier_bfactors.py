#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:45:00 2022

@author: alok
"""
#%% Introduction
'''
This script is used to find the effect of scattering atoms on the local bfactor correlation plots

Input: 
    1) refined  atomic model path
    2) number of scattering models (1A, 2A, 5A, 10A etc)
Output: 
    1) local bfactor correlation of atomic model map and scattered atomic model map
    2) local bfactor correlation of scattered atomic model map and atomic bfactors of scattered model
'''

import mrcfile
import gemmi
import os
import numpy as np
import random

from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask
from locscale.include.emmer.pdb.pdb_tools import get_all_atomic_positions, get_atomic_bfactor_window, find_wilson_cutoff
from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position
from locscale.include.emmer.ndimage.map_tools import get_local_bfactor_emmap
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array, estimate_bfactor_standard
from locscale.utils.plot_utils import plot_correlations, plot_correlations_multiple
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



#%% Inputs
local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

print("Preparing Figure 2(f)")

pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")
emmap_path = os.path.join(input_folder, "emd_10692_additional_1.map")
window_size_A = 25
sample_size = 1000

emmap = mrcfile.open(emmap_path).data
apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]

window_size_pix = int(round(window_size_A/apix))

st = gemmi.read_structure(pdb_path)
atomic_positions = list(get_all_atomic_positions(gemmi_structure=st, as_dictionary=False))

pdb_centers = random.sample(atomic_positions, sample_size)
mrc_centers = convert_pdb_to_mrc_position(pdb_centers, apix=apix)

atomic_bfactors = []
guinier_bfactors = []
for i,pdb_center in enumerate(tqdm(pdb_centers, desc="Analysing bfactors")):
    mrc_center = mrc_centers[i]
    
    atomic_bfactor = get_atomic_bfactor_window(st, pdb_center, window_size_A)
    guinier_bfactor = get_local_bfactor_emmap(emmap_path, mrc_center, fsc_resolution=2.8, boxsize=window_size_pix, 
                                              wilson_cutoff="traditional")[0]
    
    atomic_bfactors.append(atomic_bfactor)
    guinier_bfactors.append(guinier_bfactor)
    

atomic_bfactors = np.array(atomic_bfactors)
guinier_bfactors = np.array(guinier_bfactors)

select_indices = np.where(atomic_bfactors > 40)
atomic_bfactors_filtered = atomic_bfactors[select_indices]
guinier_bfactors_filtered = guinier_bfactors[select_indices]

plot_correlations(atomic_bfactors_filtered, guinier_bfactors_filtered, scatter=True, 
                  x_label="Local Atomic B-factors $\AA^{2}$",
                  y_label="Local Wilson B-factors $\AA^{2}$",
                  output_folder=output_folder,
                  filename="Figure_2f_atomic_v_wilson_bfactors_scatter_perturb_0A_using_emmap.eps")

    
    