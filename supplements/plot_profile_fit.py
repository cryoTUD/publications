#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:38:58 2022

@author: alok
"""
## Script to plot bfactor at an atom location in the map

import os
from pathlib import Path
import gemmi
import mrcfile
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, plot_radial_profile, frequency_array, estimate_bfactor_standard
from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position, extract_window
from locscale.utils.plot_utils import pretty_plot_radial_profile
import numpy as np

local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

print("Preparing Supplementary Figure 2")
emmap_path = os.path.join(input_folder, "emd_10692_additional_1.map")
pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")

window_size_A = 25

apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]
window_size_pix = int(round(window_size_A/apix))

wilson_cutoff = 10 ## Use traditional approach
fsc_resolution = 2.8

## Atom spec
chain_name_1 = "C"
res_seqid_1 = 67
atom_name_1 = "CZ"

chain_name_2 = "A"
res_seqid_2 = 303
atom_name_2 = "CA"



## Calculation begin

# Get pdb coordinate

def get_emmap_bfactor_fit_to_pdb(emmap_path, pdb_path, wilson_cutoff, fsc_cutoff):
    from tqdm import tqdm
    
    st = gemmi.read_structure(pdb_path)
    st_clone = st.clone()
    emmap = mrcfile.open(emmap_path).data
    apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]
    for cra_obt in tqdm(st_clone[0].all()):
        atom = cra_obt.atom
        atom_position = atom.pos.tolist()
        mrc_position = convert_pdb_to_mrc_position([atom_position], apix)[0]
        emmap_wn = extract_window(emmap, mrc_position, size=window_size_pix)
        rp_emmap_wn = compute_radial_profile(emmap_wn)
        freq = frequency_array(rp_emmap_wn, apix)
        bfactor, amp, qfit = estimate_bfactor_standard(freq, rp_emmap_wn, wilson_cutoff=wilson_cutoff, fsc_cutoff=fsc_cutoff, return_amplitude=True, return_fit_quality=True, standard_notation=True)
        
        local_qfit = round(qfit,2)
        local_bfactor = bfactor
        atom.occ = local_qfit
        atom.b_iso = bfactor
    
    new_filename = pdb_path[:-4]+"_replaced_occupancy_with_qfit.pdb"
    
    st_clone.write_pdb(new_filename)
        
    
    
def get_local_profile_fit(emmap_path, pdb_path, chain_name, res_seqid, atom_name, wilson_cutoff,fsc_cutoff):
    st = gemmi.read_structure(pdb_path)
    emmap = mrcfile.open(emmap_path).data
    apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]
    for res in st[0][chain_name]:
        if res.seqid.num == res_seqid:
            for atom in res:
                if atom.name == atom_name:
                    atom_position = atom.pos.tolist()
    
    mrc_position = convert_pdb_to_mrc_position([atom_position], apix)[0]
    print(mrc_position)
    emmap_wn = extract_window(emmap, mrc_position, size=window_size_pix)
    
    rp_emmap_wn = compute_radial_profile(emmap_wn)
    freq = frequency_array(rp_emmap_wn, apix)
    
    bfactor, amp, qfit = estimate_bfactor_standard(freq, rp_emmap_wn, wilson_cutoff=wilson_cutoff, fsc_cutoff=fsc_cutoff, return_amplitude=True, return_fit_quality=True, standard_notation=True)
    exponential_fit = amp * np.exp(-0.25 * bfactor * freq**2)
    
    profiles = {
        'freq':freq,
        'rp':rp_emmap_wn,
        'exp':exponential_fit,
        'qfit':round(qfit,2)}
    
    return profiles



profiles_1 = get_local_profile_fit(emmap_path, pdb_path, chain_name_1, res_seqid_1, atom_name_1, wilson_cutoff, fsc_resolution)    
profiles_2 = get_local_profile_fit(emmap_path, pdb_path, chain_name_2, res_seqid_2, atom_name_2, wilson_cutoff, fsc_resolution)    

fig_1 = pretty_plot_radial_profile(profiles_1['freq'], [profiles_1['exp'], profiles_1['rp']], legends=["$R^2$: {}".format(profiles_1['qfit'])], 
                                   squared_amplitudes=True, normalise=False, ylims=[-5,10], fontscale=3, linewidth=2, logScale=True)
fig_2 = pretty_plot_radial_profile(profiles_2['freq'], [profiles_2['exp'], profiles_2['rp']], legends=["$R^2$: {}".format(profiles_2['qfit'])], 
                                   squared_amplitudes=True, normalise=False, ylims=[-5,10], fontscale=3, linewidth=2, logScale=True)

filename_1 = os.path.join(output_folder, "Figure_S2_profile_fit_at_{}_{}_{}_version8.eps".format(chain_name_1, res_seqid_1, atom_name_1,profiles_1['qfit']))
filename_2 = os.path.join(output_folder, "Figure_S2_profile_fit_at_{}_{}_{}_version8.eps".format(chain_name_2, res_seqid_2, atom_name_2))

fig_1.savefig(filename_1, dpi=600, bbox_inches="tight",format="eps")
fig_2.savefig(filename_2, dpi=600, bbox_inches="tight",format="eps")



#%%
#get_emmap_bfactor_fit_to_pdb(emmap_path, pdb_path,wilson_cutoff, fsc_resolution)

