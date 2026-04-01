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


import os
import numpy as np

from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask, shake_pdb
from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask


#%% Inputs
folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs/"
output_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs"
refined_atomic_model_path = os.path.join(folder, "pdb6y5a_additional_refined.pdb")
sample_mask_path = os.path.join(folder,"emd_10692_additional_1_confidenceMap.mrc")


rmsd_magnitudes = np.array([0.1,0.2,1, 2, 5, 10, 15, 20])
#rmsd_magnitudes = np.array([2,5])

#%% Calculation

model_mask_path = get_atomic_model_mask(emmap_path=sample_mask_path, pdb_path=refined_atomic_model_path, dilation_radius=3)



shaken_structures = {}
for rmsd_magnitude in rmsd_magnitudes:
    shaken_structures[rmsd_magnitude] = shake_pdb_within_mask(refined_atomic_model_path, model_mask_path, rmsd_magnitude, masking="strict")
    #shaken_structures[rmsd_magnitude] = shake_pdb(refined_atomic_model_path, rmsd_magnitude)
    filename = os.path.join(output_folder, "pdb6y5a_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_pm.pdb".format(int(rmsd_magnitude*100)))
    shaken_structures[rmsd_magnitude].write_pdb(filename)

