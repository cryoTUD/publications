#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:02:08 2022

@author: alok
"""
import os
import mrcfile
from locscale.include.emmer.ndimage.profile_tools import frequency_array
from locscale.include.emmer.ndimage.profile_tools import plot_radial_profile, get_theoretical_profile, scale_profiles,generate_no_debye_profile
from locscale.include.emmer.ndimage.map_tools import apply_radial_profile, set_radial_profile_to_volume, compute_radial_profile_simple
from locscale.include.emmer.ndimage.map_quality_tools import calculate_surface_area_at_threshold
from locscale.include.emmer.ndimage.map_utils import save_as_mrc
from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask, shake_pdb, set_atomic_bfactors
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
import gemmi
import numpy as np


#%% Generate helix map with no debye effect

input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Calculations/debye_effect"
output_folder = input_folder


backbone_pdb_path = os.path.join(input_folder, "pdb_b_424_461_backbone_rmsd_0_pm.pdb")


apix = 0.832
size = (320,320,320)
b_iso = 100
## Set the atomic bfactors to a common value
backbone_st = gemmi.read_structure(backbone_pdb_path)
uniform_bfactor_backbone_structure = set_atomic_bfactors(input_gemmi_st=backbone_st, b_iso=b_iso)
uniform_bfactor_backbone_structure.write_pdb(os.path.join(input_folder, "backbone_pdb_biso_{}.pdb".format(b_iso)))

# Compute the simulated map from each RMSD

proper_backbone_map = pdb2map(input_pdb=uniform_bfactor_backbone_structure, apix=apix, size=size)

save_as_mrc(proper_backbone_map, output_filename=os.path.join(output_folder, "proper_backbone.mrc"), apix=apix)

threshold = 0.05
# Get a no debye profile at the right bfactor
rp_proper_backbone= compute_radial_profile_simple(proper_backbone_map)
freq = frequency_array(rp_proper_backbone, apix)
#%%
#cutoffs = np.linspace(20,2,6,dtype=int)
cutoffs = [10,8,6,4,2]
no_debye_profile = {}
surface_area_threshold = {}
for wilson_cutoff in cutoffs:
    no_debye_profile[wilson_cutoff] = generate_no_debye_profile(freq, rp_proper_backbone, wilson_cutoff=wilson_cutoff)
    no_debye_map = set_radial_profile_to_volume(proper_backbone_map, no_debye_profile[wilson_cutoff])
    surface_area_threshold[wilson_cutoff] = calculate_surface_area_at_threshold(no_debye_map, apix, reference_threshold=threshold)
    save_as_mrc(no_debye_map, output_filename=os.path.join(output_folder, "no_debye_map_wilson_cutoff_{}A.mrc".format(wilson_cutoff)),apix=apix)

# Apply the helix and sheet profile to the "no-debye" backbone map

#helix_theoretical_profile = get_theoretical_profile(length=len(rp_proper_backbone), apix=apix, profile_type="helix")[1]
#sheet_theoretical_profile = get_theoretical_profile(length=len(rp_proper_backbone), apix=apix, profile_type="sheet")[1]

## Scale theoretical profiles to the right bfactor
'''
#scaled_helix_profile = scale_profiles((freq,no_debye_profile), (freq,helix_theoretical_profile), wilson_cutoff=4, fsc_cutoff=2)[1]
scaled_sheet_profile = scale_profiles((freq,no_debye_profile), (freq,sheet_theoretical_profile), wilson_cutoff=4, fsc_cutoff=2)[1]

backbone_map_no_debye_with_helix_profile = set_radial_profile_to_volume(no_debye_map, scaled_helix_profile)
backbone_map_no_debye_with_sheet_profile = set_radial_profile_to_volume(no_debye_map, scaled_sheet_profile)

save_as_mrc(backbone_map_no_debye_with_helix_profile, os.path.join(output_folder,"repaired_with_helix_profile.mrc"), apix=apix)
save_as_mrc(backbone_map_no_debye_with_sheet_profile, os.path.join(output_folder,"repaired_with_sheet_profile.mrc"), apix=apix)


rp_repaired_map_helix = compute_radial_profile(backbone_map_no_debye_with_helix_profile)
'''

plot_radial_profile(freq, list(no_debye_profile.values())+[rp_proper_backbone], legends=["${} \AA$".format(x) for x in cutoffs]+["backbone"], logScale=True, showPoints=False)


#%%
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile
from locscale.include.emmer.ndimage.map_tools import compute_radial_profile_simple
#rp_old = compute_radial_profile(proper_backbone_map)
#rp_new = compute_radial_profile_simple(proper_backbone_map)[:-1]

#plot_radial_profile(freq, [rp_old,rp_new], legends=["Old function","New function"])