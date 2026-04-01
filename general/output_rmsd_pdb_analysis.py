#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:07:51 2022

@author: alok
"""
import os
from locscale.include.emmer.pdb.pdb_utils import compute_rmsd_two_pdb
from locscale.utils.plot_utils import pretty_violinplots, pretty_boxplots
import matplotlib.pyplot as plt

rmsd_magnitudes = [0,100,200,500,1000,1500,2000]


input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Calculations/effect_of_strict_rmsd"
strict_rmsd_prefix = "pdb6y5a_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_pm.pdb"
rough_rmsd_prefix = "pdb6y5a_additional_refined_perturbed_rmsd_{}_pm.pdb"

strict_rmsd_paths = {}
rough_rmsd_paths = {}
for rmsd in rmsd_magnitudes:
    strict_rmsd_paths[rmsd] = os.path.join(input_folder, strict_rmsd_prefix.format(rmsd))
    rough_rmsd_paths[rmsd] = os.path.join(input_folder, rough_rmsd_prefix.format(rmsd))
   

strict_rmsd_distances = {}
rough_rmsd_distances = {}

for rmsd in rmsd_magnitudes[1:]:
    strict_rmsd_distances[rmsd] = compute_rmsd_two_pdb(input_pdb_1=strict_rmsd_paths[rmsd], input_pdb_2=strict_rmsd_paths[0],return_array=True)
    rough_rmsd_distances[rmsd] = compute_rmsd_two_pdb(input_pdb_1=rough_rmsd_paths[rmsd], input_pdb_2=rough_rmsd_paths[0],return_array=True)
    
strict_rmsd_filename = os.path.join(input_folder, "strict_rmsd_rmsd.svg")
rough_rmsd_filename = os.path.join(input_folder, "rough_rmsd_rmsd.svg")

pretty_boxplots(list(strict_rmsd_distances.values()), 
                   xticks=["{} $\AA$".format(int(x/100)) for x in list(strict_rmsd_distances.keys())],
                   ylabel="Output RMSD ($\AA$)", xlabel="Imposed RMSD ($\AA$)",filename=strict_rmsd_filename)


pretty_boxplots(list(rough_rmsd_distances.values()), 
                   xticks=["{} $\AA$".format(int(x/100)) for x in list(rough_rmsd_distances.keys())],
                   ylabel="Output RMSD ($\AA$)", xlabel="Imposed RMSD ($\AA$)",filename=rough_rmsd_filename)
    
    
    