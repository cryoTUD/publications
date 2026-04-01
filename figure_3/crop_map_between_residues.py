#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:58:13 2021

@author: alok
"""

## Script to crop part of the map for visualisation

import os
import mrcfile
from locscale.include.emmer.ndimage.map_tools import crop_map_between_residues
from locscale.include.emmer.ndimage.map_utils import save_as_mrc


input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs_11"
output_folder = input_folder #"/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/local_resolution_crop"

pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")

unsharpened_map_path = os.path.join(input_folder, "emd_10692_additional_1.map")
globally_sharpened_map_path = os.path.join(input_folder, "emd_10692_global_sharpened_filtered.mrc")
bfactor_map_paths = [os.path.join(input_folder, "bfactor_map_rmsd_{}.mrc".format(x)) for x in [0,1,2,5,10,15,20]]
#bfactor_map_paths = [os.path.join(input_folder, "processing_files","bfactor_map.mrc")]
#apix = mrcfile.open(bfactor_map_paths[0]).voxel_size.tolist()[0]
apix=0.832
corrected_bfactor_map_path = os.path.join(input_folder, "bfactor_map_modelmask_10_to_fsc.mrc")
mask_map_path = os.path.join(input_folder, "emd_10692_additional_1_confidenceMap.mrc")
locscale_map_paths = [os.path.join(input_folder, "locscale_modelmask_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_A.mrc".format(x)) for x in [0,1,2,5,10,15,20]]
locscale_map_path_correct_bfactor_perturbed_pdb = os.path.join(input_folder, "locscale_atomic_model_mask_correct_bfactor_10_to_FSC.mrc")
local_resolution_map_path = os.path.join(input_folder, "emd_10692_half_map_1_localResolutions.mrc")
locscale_randomised_bfactor_path = os.path.join(input_folder, "locscale_using_randomised_bfactor_modelmask.mrc")
randomised_bfactor_map_path = os.path.join(input_folder, "bfactor_map_randomised_bfactor.mrc")
#chain_name = "B"
#residue_range = [282, 309]

chain_name = "A"
residue_range = [250,272]

#chain_name = "D"
#residue_range = [501,501]   ## Serotonin molecule residue 

#chain_name = "A"
#residue_range = [319, 331]

#chain_name = "E"
#residue_range = [319, 331]

cropped_map = {}

#for emmap_path in [unsharpened_map_path, globally_sharpened_map_path]+locscale_map_paths:#+bfactor_map_paths:
#for emmap_path in [os.path.join(input_folder, "locscale_using_random_bfactors.mrc"),randomised_bfactor_map_path]:
#for emmap_path in [randomised_bfactor_map_path]+bfactor_map_paths:
#for emmap_path in [mask_map_path]:
#for emmap_path in [locscale_map_path_correct_bfactor_perturbed_pdb,corrected_bfactor_map_path]:
#for emmap_path in [locscale_randomised_bfactor_path,randomised_bfactor_map_path]:
#for emmap_path in [local_resolution_map_path]:
for emmap_path in locscale_map_paths+bfactor_map_paths:
    emmap_name = os.path.basename(emmap_path)
    cropped_segment = crop_map_between_residues(emmap_path=emmap_path, pdb_path=pdb_path, chain_name=chain_name, 
                                                residue_range=residue_range, dilation_radius=3)
    
    output_filename = os.path.join(output_folder, "cropped_{}_{}_{}_".format(chain_name, residue_range[0], residue_range[1])+emmap_name)
    save_as_mrc(cropped_segment, output_filename, apix=apix)

