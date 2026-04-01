#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:58:08 2022

@author: alok
"""
## Script to map out bfactor changes in the perturbed model maps
import os
import mrcfile
from locscale.utils.plot_utils import crop_data_to_map
from locscale.include.emmer.ndimage.map_utils import save_as_mrc

input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs"
output_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_2/delta_bfactor"

mask_path = os.path.join(input_folder, "emd_10692_additional_1_confidenceMap.mrc")
bfactor_rmsd_0 = os.path.join(input_folder, "bfactor_map_rmsd_0.mrc")
apix = mrcfile.open(mask_path).voxel_size.tolist()[0]

bfactor_rmsd_path = {}
rmsd_magnitudes = [0,100,200,500,1000,1500,2000]

for rmsd in rmsd_magnitudes:
    bfactor_rmsd_path[rmsd] = os.path.join(input_folder, "bfactor_map_rmsd_{}.mrc".format(int(rmsd/100)))

bfactor_arrays = {}
for rmsd in rmsd_magnitudes:
    bfactor_arrays[rmsd] = mrcfile.open(bfactor_rmsd_path[rmsd]).data

delta_bfactor_rmsd = {}
for rmsd in rmsd_magnitudes[1:]:
    delta_bfactor_rmsd[rmsd] = bfactor_arrays[rmsd]-bfactor_arrays[0]

    save_as_mrc(map_data=delta_bfactor_rmsd[rmsd], 
                output_filename=os.path.join(output_folder, "delta_bfactor_{}_to_0_rmsd.mrc".format(int(rmsd/100))), apix=apix)




#%%
import seaborn as sns

bfactor_array_rmsd_0 = crop_data_to_map(input_data_map=bfactor_arrays[0], mask=mrcfile.open(mask_path).data, mask_threshold=0.5)

sns.kdeplot(bfactor_array_rmsd_0)

sns.kdeplot(crop_data_to_map(input_data_map=bfactor_arrays[200], mask=mrcfile.open(mask_path).data, mask_threshold=0.5))

sns.kdeplot(crop_data_to_map(input_data_map=bfactor_arrays[2000], mask=mrcfile.open(mask_path).data, mask_threshold=0.5))
