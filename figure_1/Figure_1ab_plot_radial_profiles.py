#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:08:04 2022

@author: alok
"""
import os 
import mrcfile
import gemmi
from locscale.include.emmer.pdb.pdb_utils import set_atomic_bfactors
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
from locscale.include.emmer.ndimage.map_utils import save_as_mrc
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array
from locscale.utils.plot_utils import pretty_plot_radial_profile

print("Plotting Figures 1(a) and 1(b)... \nPlease wait (approximately 7 mins)")
local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

#pdb_path = os.path.join(input_folder, "pdb_b_424_461_backbone.pdb")

## Simulation parameters
apix = 0.5
unitcell = gemmi.UnitCell(320*0.5,320*0.5,320*0.5, 90, 90, 90)


#st = gemmi.read_structure(pdb_path)
uniform_bfactor_range = [0,50,100,150,200,250,300]

uniform_bfactor_structures = {}
model_map_bfactor = {}
model_map_bfactor_backbone = {}
radial_profile_bfactor = {}
radial_profile_bfactor_backbone = {}
for bfactor in uniform_bfactor_range:
    model_map_bfactor[bfactor] = mrcfile.open(os.path.join(input_folder, "full_side_chain","uniform_bfactor_{}_0p5.mrc".format(bfactor))).data
    model_map_bfactor_backbone[bfactor] = mrcfile.open(os.path.join(input_folder, "backbone_only","backbone_uniform_bfactor_{}_0p5.mrc".format(bfactor))).data
    
    radial_profile_bfactor[bfactor] = compute_radial_profile(model_map_bfactor[bfactor])
    radial_profile_bfactor_backbone[bfactor] = compute_radial_profile(model_map_bfactor[bfactor])
    freq = frequency_array(radial_profile_bfactor[bfactor], apix=apix)


#%%
fig=pretty_plot_radial_profile(freq, list(radial_profile_bfactor.values()), 
                    showlegend=False, normalise=False, squared_amplitudes=True, logScale=True, ylims=[0,20],crop_freq=[100,2], linewidth=2, fontscale=4)


fig.savefig(os.path.join(output_folder, "Figure_1a_sidechain_radial_profiles.eps"), dpi=600, bbox_inches="tight",format="eps")

backbonefig=pretty_plot_radial_profile(freq, list(radial_profile_bfactor_backbone.values()), 
                    showlegend=False, normalise=True, squared_amplitudes=False, logScale=False, ylims=[-0.005,0.2],crop_freq=[100,2], linewidth=2, fontscale=4)


backbonefig.savefig(os.path.join(output_folder, "Figure_1b_backbone_radial_profiles.eps"), dpi=600, bbox_inches="tight",format="eps")


