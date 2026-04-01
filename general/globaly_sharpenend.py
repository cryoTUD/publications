#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 23:06:31 2022

@author: alok
"""
## Script to obtain FSC weighted globally sharpened map

import os
from locscale.include.emmer.ndimage.fsc_util import apply_fsc_filter
from locscale.include.emmer.ndimage.map_utils import save_as_mrc
import mrcfile
input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/global_sharpened_map"

globally_sharpened_map_path = os.path.join(input_folder, "emd_10692_additional_1_global_sharpened.mrc")
halfmap1_path = os.path.join(input_folder, "emd_10692_half_map_1.map")
halfmap2_path = os.path.join(input_folder, "emd_10692_half_map_2.map")

globally_sharpened_map = mrcfile.open(globally_sharpened_map_path).data
halfmap1 = mrcfile.open(halfmap1_path).data
halfmap2 = mrcfile.open(halfmap2_path).data

apix = mrcfile.open(globally_sharpened_map_path).voxel_size.tolist()[0]

filtered_map,_ = apply_fsc_filter(emmap=globally_sharpened_map, apix=apix, halfmap_1=halfmap1, halfmap_2=halfmap2)

save_as_mrc(filtered_map, output_filename=os.path.join(input_folder, "fsc_filtered_globally_sharpened.mrc"),apix=apix)