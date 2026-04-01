#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:38:58 2022

@author: alok
"""
## Script to plot bfactor at an atom location in the map

import os
import gemmi
import mrcfile
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array, estimate_bfactor_standard
from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position, extract_window
import numpy as np

local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

print("Preparing Figure 2(g)")
emmap_path = os.path.join(input_folder, "emd_10692_additional_1.map")
local_resolution_map_path = os.path.join(input_folder, "emd_10692_half_map_1_localResolutions.mrc")
pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")

window_size_A = 25

apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]
window_size_pix = int(round(window_size_A/apix))

wilson_cutoff = 10 ## Use traditional approach
fsc_resolution = 2.8

## Atom spec
chain_name_1 = "C"
res_seqid_1 = 67
atom_name_1 = "CA"

## Atom spec
chain_name_2 = "B"
res_seqid_2 = 443
atom_name_2 = "CA"

## Atom spec
chain_name_3 = "E"
res_seqid_3 = 332
atom_name_3 = "CA"
## Calculation begin




# Get pdb coordinate
def get_local_profile(emmap_path, pdb_path, chain_name, res_seqid, atom_name, wilson_cutoff,fsc_cutoff, local_resolution_path):
    st = gemmi.read_structure(pdb_path)
    emmap = mrcfile.open(emmap_path).data
    apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]
    local_resolution_data = mrcfile.open(local_resolution_path).data
    
    for res in st[0][chain_name]:
        if res.seqid.num == res_seqid:
            ca_atom_position = res.get_ca().pos.tolist()
    
    mrc_position = convert_pdb_to_mrc_position([ca_atom_position], apix)[0]
    
    emmap_wn = extract_window(emmap, mrc_position, size=window_size_pix)
    local_resolution_point = local_resolution_data[mrc_position[0], mrc_position[1],mrc_position[2]]
    
    rp_emmap_wn = compute_radial_profile(emmap_wn)
    freq = frequency_array(rp_emmap_wn, apix)
    
    bfactor, amp, qfit = estimate_bfactor_standard(freq, rp_emmap_wn, wilson_cutoff=wilson_cutoff, fsc_cutoff=fsc_cutoff, return_amplitude=True, return_fit_quality=True, standard_notation=True)
    exponential_fit = amp * np.exp(-0.25 * bfactor * freq**2)
    print("Local Resolution: {:.2f}".format(local_resolution_point))
    print("Local B-factor: {:.2f}".format(bfactor))
    profiles = {
        'freq':freq,
        'rp':rp_emmap_wn,
        'exp':exponential_fit,
        'qfit':round(qfit,2),
        'fsc':round(local_resolution_point,2)}
    
    return profiles


profiles_1 = get_local_profile(emmap_path, pdb_path, chain_name_1, res_seqid_1, atom_name_1, wilson_cutoff, fsc_resolution,local_resolution_map_path)    
profiles_2 = get_local_profile(emmap_path, pdb_path, chain_name_2, res_seqid_2, atom_name_2, wilson_cutoff, fsc_resolution,local_resolution_map_path)    
profiles_3 = get_local_profile(emmap_path, pdb_path, chain_name_3, res_seqid_3, atom_name_3, wilson_cutoff, fsc_resolution,local_resolution_map_path)    

#%%
from locscale.utils.plot_utils import pretty_plot_radial_profile
fig_1 = pretty_plot_radial_profile(profiles_1['freq'], [profiles_1['rp'],profiles_2['rp'],profiles_3['rp']], 
                                   legends=["$FSC-FDR$: {:.1f}".format(profiles_1['fsc']),"$FSC-FDR$: {:.1f}".format(profiles_2['fsc']),"$FSC-FDR$: {:.1f}".format(profiles_3['fsc'])], 
                                   logScale=False, ylims=[-0.005,0.2], yticks=[0,0.1,0.2], fontscale=3, linewidth=2)

fig_2 = pretty_plot_radial_profile(profiles_1['freq'], [profiles_1['rp'],profiles_2['rp'],profiles_3['rp']],figsize=(7,4), 
                                   legends=["$FSC$: {:.1f}".format(profiles_1['fsc']),"$FSC$: {:.1f}".format(profiles_2['fsc']),"$FSC-FDR$: {:.1f}".format(profiles_3['fsc'])], 
                                   logScale=True, normalise=False, showlegend=False,fontscale=2, linewidth=2, crop_freq=[3.5,1.8])

filename_1 = os.path.join(output_folder, "Figure_2g_combined_local_profiles_three_resolution_CBE.eps")
filename_2 = os.path.join(output_folder, "Figure_2g_combined_local_profiles_three_resolution_CBE_crop_small_figsize.eps")
fig_1.savefig(filename_1, dpi=600, bbox_inches="tight", format="eps")
fig_2.savefig(filename_2, dpi=600, bbox_inches="tight", format="eps")




