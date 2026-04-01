#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:22:41 2022

@author: alok
"""
import pickle
import pypdb
import os
from tqdm import tqdm
input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/Input"
output_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_3/Output"


radial_profiles_file = os.path.join(input_folder,'secondary_structure_profiles.pickle')
radial_profiles_file_2 = os.path.join(input_folder,'nucleotide_profiles.pickle')

with open(radial_profiles_file,'rb') as f:
    secondary_structure_profile_raw = pickle.load(f)
        
with open(radial_profiles_file_2,'rb') as f:
    nucleotide_profile_raw = pickle.load(f)    
    
unitcell_size = {}
for pdbid in tqdm(secondary_structure_profile_raw.keys(), desc="Secondary structures"):
    try:
        pdb_info = pypdb.get_all_info(pdbid)
        max_cell_size = max(pdb_info['cell']['length_a'],pdb_info['cell']['length_b'],pdb_info['cell']['length_c'])
        unitcell_size[pdbid] = max_cell_size
    except:
        continue

unitcell_rna = {}
unitcell_dna = {}
for index in tqdm(nucleotide_profile_raw.keys(),desc="Nucleotides"):
    rna_id = nucleotide_profile_raw[index]['rna_id']
    try:
        pdb_info = pypdb.get_all_info(rna_id)
        max_cell_size = max(pdb_info['cell']['length_a'],pdb_info['cell']['length_b'],pdb_info['cell']['length_c'])
        unitcell_rna[rna_id] = max_cell_size
    except:
        continue
    dna_id = nucleotide_profile_raw[index]['dna_id']
    try:
        pdb_info = pypdb.get_all_info(dna_id)
        max_cell_size = max(pdb_info['cell']['length_a'],pdb_info['cell']['length_b'],pdb_info['cell']['length_c'])
        unitcell_dna[dna_id] = max_cell_size
    except:
        continue
    