#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:38:35 2022

@author: alok
"""

## Figure 2 Plots

## Plots required in this figure: FSC and RSCC between Locscale with perturb equal to upto 20A 

import os
import mrcfile
import numpy as np
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
from locscale.include.emmer.ndimage.profile_tools import frequency_array
from locscale.include.emmer.ndimage.map_tools import compute_real_space_correlation, get_atomic_model_mask
from locscale.include.emmer.pdb.pdb_utils import set_atomic_bfactors
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
from tqdm import tqdm
import gemmi
import matplotlib.pyplot as plt

## Inputs
input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs"
output_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_2/fsc_rscc"
locscale_output_prefix = "locscale_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_A.mrc"
pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")
unsharpened_emmap_path = os.path.join(input_folder, "emd_10692_additional_1.map")
#mask_path = os.path.join(input_folder, "emd_10692_additional_1_confidenceMap.mrc")    ## uncomment if you need FDR mask
#mask_data = mrcfile.open(mask_path).data

rmsd_magnitudes = [0,1,2,5,10,15,20]
locscale_perturb_map_paths = {}
for rmsd in rmsd_magnitudes:
    locscale_perturb_map_paths[rmsd] = os.path.join(input_folder, locscale_output_prefix.format(rmsd))

## Get a softmask from either model mask or FDR mask
model_mask_path = get_atomic_model_mask(unsharpened_emmap_path, pdb_path)
softmask = mrcfile.open(model_mask_path).data

#binarised_mask = (mask_data>=0.99).astype(np.int_)
#softmask = get_cosine_mask(binarised_mask, length_cosine_mask_1d=7)

## Get inital data

apix = mrcfile.open(locscale_perturb_map_paths[0]).voxel_size.tolist()[0]
## get EMmap
unsharpened_emmap = mrcfile.open(unsharpened_emmap_path).data
## Set to zero bfactor map
st = gemmi.read_structure(pdb_path)
st_b0 = set_atomic_bfactors(input_gemmi_st=st, b_iso=0)
simmap_bfactor0 = pdb2map(st_b0, apix=apix, size=unsharpened_emmap.shape)
simmap_refined = pdb2map(st, apix=apix, size=unsharpened_emmap.shape)

## Calculation

fsc_arrays_perturb = {}
rscc_perturb = {}
for rmsd in tqdm(rmsd_magnitudes, desc="Calculating map statistics"):
    input_map1 = mrcfile.open(locscale_perturb_map_paths[0]).data * softmask    #Uncomment if you need to use locscale RMSD 0 map
    #input_map1 = unsharpened_emmap * softmask ## Uncomment if you need to use unsharpened emmap
    #input_map1 = simmap_bfactor0  * softmask ## Uncomment if you need to use zero bfactor simmap
    #input_map1 = simmap_refined  * softmask## Uncomment if you need to use Refined bfactor simmap
    input_map2 = mrcfile.open(locscale_perturb_map_paths[rmsd]).data * softmask
    
    # Calculate FSC and RSCC
    fsc = calculate_fsc_maps(input_map_1=input_map1, input_map_2=input_map2)
    rscc = compute_real_space_correlation(input_map_1=input_map1, input_map_2=input_map2)
    freq = frequency_array(fsc, apix=apix)
    
    # Add results to an array
    fsc_arrays_perturb[rmsd] = [freq,fsc]
    rscc_perturb[rmsd] = rscc


#%% Plotting
from locscale.utils.plot_utils import pretty_lineplot_multiple_fsc_curves

fsc_filename = os.path.join(output_folder, "Figure_2_FSC_locscale_perturbed_rmsd_0.eps")

legends = ["{} $\AA$".format(x) for x in rmsd_magnitudes]
pretty_lineplot_multiple_fsc_curves(fsc_arrays_perturb, filename=fsc_filename, fontscale=3, linewidth=2, legends=legends)


#%%%

plt.figure()
from locscale.utils.plot_utils import pretty_lineplot_XY
rscc_filename = os.path.join(output_folder, "Figure_2_RSCC_locscale_simmap_refined_bfactor.svg")
pretty_lineplot_XY(list(rscc_perturb.keys()), list(rscc_perturb.values()), "RMSD ($\AA$)", "Real Space Correlation", filename=rscc_filename,
                   linewidth=2, marker="o",markersize=12, fontscale=3)
    


