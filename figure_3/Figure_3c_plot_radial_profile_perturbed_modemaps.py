#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:16:17 2022

@author: alok
"""


## Script to plot radial profiles of scattered model maps

from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile_from_mrcs, compute_radial_profile, frequency_array, plot_radial_profile
from locscale.include.emmer.ndimage.map_tools import compute_radial_profile_simple
from locscale.utils.plot_utils import pretty_plot_radial_profile
import os
import seaborn as sns
import mrcfile

local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

print("Preparing Figure 3(c)...")
rmsd_magnitudes = [0,1,2,5,10,15, 20]

modelmap_name_prefix = "pdb6y5a_modelmap_no_overlap_rmsd_{}.mrc"
radial_profiles = {}
for rmsd in rmsd_magnitudes:
    print("RMSD: {} A".format(rmsd))
    mrc_path = os.path.join(input_folder,"model_maps",modelmap_name_prefix.format(int(rmsd*100)))
    emmap = mrcfile.open(mrc_path).data
    apix = mrcfile.open(mrc_path).voxel_size.tolist()[0]
    
    rp_emmap = compute_radial_profile_simple(emmap)
    rp_emmap_norm = rp_emmap/rp_emmap.max()
    freq = frequency_array(rp_emmap, apix)
    
    radial_profiles[rmsd] = rp_emmap_norm


filename = os.path.join(output_folder, "Figure_3c_radial_profiles_modelmap.eps")
#%%
fig=pretty_plot_radial_profile(freq, list(radial_profiles.values()), normalise=True, legends=["{} $\AA$".format(x) for x in rmsd_magnitudes],
                           logScale=False, showPoints=False,crop_freq=[200,2.5], ylims=[-0.001,0.01],
                           linewidth=1,fontscale=1.75, figsize=(7,5))

fig.savefig(filename, dpi=600, bbox_inches="tight",format="eps")