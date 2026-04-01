#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:53:15 2022

@author: alok
"""
'''
This script is to generate multiple PDBs where the occupancy metrics is replaced by the distance of the atom to an unperturbed model
'''

import os
from locscale.include.emmer.pdb.pdb_utils import replace_pdb_column_with_arrays, compute_rmsd_two_pdb


folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/inputs"

unperturbed_pdb_path = os.path.join(folder, "pdb6y5a_additional_refined_perturbed_strict_masking_no_overlap_rmsd_0_pm.pdb")
rmsd_magnitudes = [100, 200, 500, 1000, 1500, 2000]  ## in picometers

perturbed_rmsd_paths = {}
for rmsd in rmsd_magnitudes:
    perturbed_rmsd_paths[rmsd] = os.path.join(folder, "pdb6y5a_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_pm.pdb".format(rmsd))


## Get distance arrays between perturbed pdb and unperturbed pdb
distance_arrays_rmsd = {}
for rmsd in rmsd_magnitudes:
    distance_arrays_rmsd[rmsd] = compute_rmsd_two_pdb(input_pdb_1=unperturbed_pdb_path, input_pdb_2=perturbed_rmsd_paths[rmsd], return_array=True)

## Replace occupancy values with distances
replaced_pdb_occupancy_with_distance_rmsd = {}
for rmsd in rmsd_magnitudes:
    replaced_pdb_occupancy_with_distance_rmsd[rmsd] = replace_pdb_column_with_arrays(input_pdb=perturbed_rmsd_paths[rmsd], 
                                                                                     replace_column="occ", replace_array=distance_arrays_rmsd[rmsd])
    
    replaced_pdb_occupancy_with_distance_rmsd[rmsd].write_pdb(os.path.join(folder, "pdb6y5a_rmsd_{}_pm_occupancy_to_distance.pdb".format(rmsd)))
