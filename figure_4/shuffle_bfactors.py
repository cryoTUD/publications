#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:26:41 2022

@author: alok
"""
## Script to randomise bfactors

import gemmi
import os
import random

input_folder = "/mnt/c/Users/abharadwaj1/Downloads/ForUbuntu/faraday_discussions/Figures/Figure_2/randomise_bfactor"
output_folder = input_folder

pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")

st = gemmi.read_structure(pdb_path)

bfactor_list = []
for cra_obj in st[0].all():
    bfactor = cra_obj.atom.b_iso
    bfactor_list.append(bfactor)


## randomise bfactors

st_copy = st.clone()
bfactor_list_copy = bfactor_list.copy()

random.shuffle(bfactor_list_copy)
for i,cra_obj_copy in enumerate(st_copy[0].all()):
    cra_obj_copy.atom.b_iso = bfactor_list_copy[i]

st_copy.write_pdb(os.path.join(output_folder,"pdb6y5a_shuffled_bfactor.pdb"))

print(st[0][0][0][0].b_iso, st_copy[0][0][0][0].b_iso)