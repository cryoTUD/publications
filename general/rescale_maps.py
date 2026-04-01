#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:31:32 2022

@author: alok
"""
## Script to rescale maps so that average DC power is the same

from locscale.include.emmer.ndimage.map_tools import compute_radial_profile_simple
from locscale.include.emmer.ndimage.map_utils import save_as_mrc
import os
import mrcfile

folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/measure_average_power"

locscale_rmsd0 = os.path.join(folder,"cropped_A_250_272_locscale_modelmask_additional_refined_perturbed_strict_masking_no_overlap_rmsd_0_A.mrc")
locscale_rmsd2 = os.path.join(folder,"cropped_A_250_272_locscale_modelmask_additional_refined_perturbed_strict_masking_no_overlap_rmsd_2_A.mrc")
locscale_rmsd20 = os.path.join(folder,"cropped_A_250_272_locscale_modelmask_additional_refined_perturbed_strict_masking_no_overlap_rmsd_20_A.mrc")

locscale_randomisedB = os.path.join(folder,"cropped_A_250_272_locscale_using_randomised_bfactor_modelmask.mrc")
locscale_correctedB = os.path.join(folder,"cropped_A_250_272_locscale_atomic_model_mask_correct_bfactor_10_to_FSC.mrc")

rmsd0 = mrcfile.open(locscale_rmsd0).data
rmsd2 = mrcfile.open(locscale_rmsd2).data
rmsd20 = mrcfile.open(locscale_rmsd20).data
randomisedB = mrcfile.open(locscale_randomisedB).data
correctedB = mrcfile.open(locscale_correctedB).data

def rescale_map(reference_map, target_map):
    
    from locscale.include.emmer.ndimage.map_tools import set_radial_profile_to_volume,compute_radial_profile_simple
    rp_reference = compute_radial_profile_simple(reference_map)
    rp_target = compute_radial_profile_simple(target_map)
    
    dc_reference = rp_reference[0]
    dc_target = rp_target[0]
    
    rp_scaled_target = dc_reference / dc_target * rp_target
    
    scaled_target_map = set_radial_profile_to_volume(target_map, rp_scaled_target)
    
    return scaled_target_map


scaled_rmsd2 = rescale_map(rmsd0, target_map=rmsd2)
scaled_rmsd20 = rescale_map(rmsd0, target_map=rmsd20)
scaled_randomisedB = rescale_map(rmsd0, target_map=randomisedB)
scaled_correctedB = rescale_map(rmsd0, target_map=correctedB)

save_as_mrc(scaled_rmsd2, "cropped_A_rescaled_locscale_rmsd2.mrc",apix=0.832)
save_as_mrc(scaled_rmsd20, "cropped_A_rescaled_locscale_rmsd20.mrc",apix=0.832)
save_as_mrc(scaled_randomisedB, "cropped_A_rescaled_locscale_randomisedB.mrc",apix=0.832)
save_as_mrc(scaled_correctedB, "cropped_A_rescaled_locscale_correctedB.mrc",apix=0.832)




    
