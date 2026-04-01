#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:02:08 2022

@author: alok
"""
import os
import mrcfile
from locscale.include.emmer.ndimage.profile_tools import frequency_array, compute_radial_profile
from locscale.include.emmer.ndimage.profile_tools import plot_radial_profile, get_theoretical_profile, scale_profiles,generate_no_debye_profile
from locscale.include.emmer.ndimage.map_tools import apply_radial_profile, set_radial_profile_to_volume, compute_radial_profile_simple
from locscale.include.emmer.ndimage.map_quality_tools import calculate_surface_area_at_threshold
from locscale.include.emmer.ndimage.map_utils import save_as_mrc
from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask, shake_pdb, set_atomic_bfactors
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
import gemmi
import numpy as np
import seaborn as sns

#%% Generate helix map with no debye effect

local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

print("Preparing Figure 5c and Suppplementary Figure 5 a")

emmap_path = os.path.join(input_folder, "apply_average_profile","proper_sidechain_b300.mrc")
modmap_path = os.path.join(input_folder,"apply_average_profile", "proper_sidechain_b50.mrc")

apix = mrcfile.open(emmap_path).voxel_size.tolist()[0]

emmap = mrcfile.open(emmap_path).data
modmap = mrcfile.open(modmap_path).data

rp_emmap = compute_radial_profile_simple(emmap)
rp_modmap = compute_radial_profile_simple(modmap)

freq = frequency_array(rp_emmap, apix)
helix_theoretical = get_theoretical_profile(len(rp_emmap), apix=apix, profile_type="helix")[1]
sheet_theoretical = get_theoretical_profile(len(rp_emmap), apix=apix, profile_type="sheet")[1]

scaled_helix_theoretical = scale_profiles((freq, rp_modmap), (freq,helix_theoretical), wilson_cutoff=10, fsc_cutoff=2)[1]
scaled_sheet_theoretical = scale_profiles((freq, rp_modmap), (freq,sheet_theoretical), wilson_cutoff=10, fsc_cutoff=2)[1]
scaled_average_theoretical = np.einsum("ij->j",np.array([scaled_helix_theoretical,scaled_sheet_theoretical]))
scaled_average_theoretical = scaled_average_theoretical/2

repaired_backbone_helix = set_radial_profile_to_volume(emmap, scaled_helix_theoretical)
repaired_backbone_sheet = set_radial_profile_to_volume(emmap, scaled_sheet_theoretical)
repaired_average_profile = set_radial_profile_to_volume(emmap, scaled_average_theoretical)

save_as_mrc(repaired_backbone_helix, os.path.join(output_folder, "repaired_helix_profile.mrc"), apix=apix)
save_as_mrc(repaired_backbone_sheet, os.path.join(output_folder,  "repaired_sheet_profile.mrc"), apix=apix)
save_as_mrc(repaired_average_profile, os.path.join(output_folder,  "repaired_average_profile.mrc"), apix=apix)

rp_repaired_helix = compute_radial_profile_simple(repaired_backbone_helix)
rp_repaired_sheet = compute_radial_profile_simple(repaired_backbone_sheet)
rp_repaired_average = compute_radial_profile_simple(repaired_average_profile)

#%%
from locscale.utils.plot_utils import pretty_plot_radial_profile

list_of_profile_analysis = [rp_emmap, rp_repaired_helix, rp_repaired_sheet,rp_repaired_average]
legends = ["B-factor: $300 \AA^{2}$",r"Average all-$\alpha$ profiles",r"Average all-$\beta$ profiles", r"Mixed $\alpha$$\beta$ profiles"]
ylims=[-0.001,0.035]

fig = pretty_plot_radial_profile(freq, list_of_profile_analysis,logScale=False, legends=legends,figsize=(20/4,12/4),showlegend=True,
                                  ylims=ylims,fontscale=5/4, linewidth=2, normalise=True, squared_amplitudes=True)

fig.savefig(os.path.join(output_folder, "Figure_5c_helix_sharpened_sidechain_version12.eps"),dpi=600,bbox_inches="tight",transparency=True)


#%%
average_theoretical = np.einsum("ij->j",np.array([helix_theoretical,sheet_theoretical]))
average_theoretical = average_theoretical/2

list_of_profile_analysis = [helix_theoretical, average_theoretical, sheet_theoretical]
legends = ["Average helix profile","Average profile","Average sheet profile"]
ylims=[-0.001,0.035]
from matplotlib.pyplot import cm
rainbow = cm.rainbow(4)
colors = rainbow[1:]
fig = pretty_plot_radial_profile(freq, list_of_profile_analysis,logScale=False, legends=legends,figsize=(20/4,12/4),showlegend=True,
                                  ylims=ylims,fontscale=5/4, linewidth=2, normalise=True, squared_amplitudes=True)

fig.savefig(os.path.join(output_folder, "Figure_S5a_average_profiles.eps"),dpi=600,bbox_inches="tight",transparency=True)

#%%
## Calculation
from tqdm import tqdm
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
fsc_arrays_perturb = {}
rscc_perturb = {}

 
#%% Plotting
from locscale.utils.plot_utils import pretty_lineplot_multiple_fsc_curves

#fsc_filename = os.path.join(output_folder, "Figure_2_FSC_locscale_perturbed_rmsd_0.eps")

legends = [r"Using all-$\alpha$-helix v all-$\beta$-sheet",r"Using all-$\alpha$-helix v mixed $\alpha$$\beta$"]
input_map1 = repaired_backbone_helix
input_map2 = repaired_backbone_sheet
input_map3 = repaired_average_profile
    # Calculate FSC and RSCC
fscab = calculate_fsc_maps(input_map_1=input_map1, input_map_2=input_map2)
fscam = calculate_fsc_maps(input_map_1=input_map1, input_map_2=input_map3)   
freq = frequency_array(fscab, apix=apix)
    
    # Add results to an array
fsc_arrays_perturb["helix-sheet"] = [freq,fscab]
fsc_arrays_perturb["helix-average"] = [freq,fscam]


#%% Plotting
from locscale.utils.plot_utils import pretty_lineplot_multiple_fsc_curves

fsc_filename = os.path.join(output_folder, "Figure_S5b_repaired.eps")

pretty_lineplot_multiple_fsc_curves(fsc_arrays_perturb, filename=fsc_filename, fontscale=3, linewidth=2, legends=legends)
