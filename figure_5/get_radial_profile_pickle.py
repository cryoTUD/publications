#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:49:22 2021

@author: alok
"""

'''
This is a python program to analyse the radial profiles from PDBs in the deposited PDB library
'''
import pickle
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
import os



def compute_and_save_radial_profiles(all_pdb_paths, save_mrc_folder, pickle_output_file,process_id):
    from locscale.include.emmer.ndimage.map_utils import save_as_mrc
    import csv
    
    apix=0.5
    size=(512,512,512)
    radial_profile_dictionary = {}
    total_pdb = len(all_pdb_paths)
    count = 0
    csvfile = open(os.path.join(save_mrc_folder,"process_id_{}.csv".format(process_id)),"w")
    writer = csv.writer(csvfile)
    for pdb_path in all_pdb_paths:
        try:
            if not os.path.exists(pdb_path):
                continue
            print("Process {} : {} percentage complete out of total {}".format(process_id, round(count/total_pdb,1), total_pdb))
            
            pdb_name = os.path.basename(pdb_path).split(".")[0]
            simmap = pdb2map(input_pdb=pdb_path, apix=apix, size=size, set_refmac_blur=True)
            map_filename = os.path.join(save_mrc_folder, pdb_name+"_map_process_{}.mrc".format(process_id))
            save_as_mrc(map_data=simmap, output_filename=map_filename, apix=apix)
            radial_profile = compute_radial_profile(simmap).tolist()
            radial_profile_dictionary[pdb_name] = {
                'amplitude': tuple(radial_profile), 
                'apix': apix}
            writer.writerow(radial_profile)
        except Exception as e:
            print("Error with pdb_path: ",pdb_path)
            print(e)
        
            
    
    
    with open(pickle_output_file,'wb') as f:
    	pickle.dump(radial_profile_dictionary,f)
    csvfile.close()



if __name__ == "__main__":
    import multiprocessing
    from locscale.utils.scaling_tools import split_sequence_evenly
    
    parent_folder = "/home/abharadwaj1/dev/studying_profiles"
    
    secondary_structure_folder = os.path.join(parent_folder, "secondary_structure")
    nucleotide_folder = os.path.join(parent_folder, "nucleotide")
        
    secondary_structure_profile_pickle = os.path.join(parent_folder, "secondary_structure_profiles.pickle")
    nucleotide_profile_pickle = os.path.join(parent_folder, "nucleotide_profiles.pickle")
    
    with open(secondary_structure_profile_pickle,'rb') as f:
        secondary_structure_profile_raw = pickle.load(f)
    
    with open(nucleotide_profile_pickle,'rb') as f:
        nucleotide_profile_raw = pickle.load(f)    
    
    
    
    ## Get PDB id of secondary structures
    
    pdb_id_secondary_structure = list(secondary_structure_profile_raw.keys())
    helix_pdb_paths = [os.path.join(secondary_structure_folder,"pdb","pdb_{}_helix_zeroB.pdb".format(p)) for p in pdb_id_secondary_structure]
    sheet_pdb_paths = [os.path.join(secondary_structure_folder,"pdb","pdb_{}_sheet_zeroB.pdb".format(p)) for p in pdb_id_secondary_structure]
    
    dna_id_list = []
    rna_id_list = []
    for index in nucleotide_profile_raw.keys():
        dna_pdbid = nucleotide_profile_raw[index]['dna_id']
        rna_pdbid = nucleotide_profile_raw[index]['rna_id']
        dna_id_list.append(dna_pdbid)
        rna_id_list.append(rna_pdbid)
    
    dna_pdb_paths = [os.path.join(nucleotide_folder,"pdb","pdb_{}_dna.pdb".format(p)) for p in dna_id_list]
    rna_pdb_paths = [os.path.join(nucleotide_folder,"pdb","pdb_{}_rna.pdb".format(p)) for p in rna_id_list]
    
    
    processes = 4
    jobs = []
    
    for i in range(processes):
        if i == 0:
            pickle_output_file = os.path.join(secondary_structure_folder, "helix_profiles_process_{}.pickle".format(i))
            save_mrc_folder = os.path.join(secondary_structure_folder, "maps","helix")

            pdb_paths_process = helix_pdb_paths
            
            process = multiprocessing.Process(target=compute_and_save_radial_profiles, 
                                              args=(pdb_paths_process,  save_mrc_folder, pickle_output_file,i))
            
            jobs.append(process)
        elif i == 1:
            pickle_output_file = os.path.join(secondary_structure_folder, "sheet_profiles_process_{}.pickle".format(i))
            save_mrc_folder = os.path.join(secondary_structure_folder, "maps","sheet")

            pdb_paths_process = sheet_pdb_paths
            
            process = multiprocessing.Process(target=compute_and_save_radial_profiles, 
                                              args=(pdb_paths_process,  save_mrc_folder, pickle_output_file,i))
            
            jobs.append(process)
        elif i == 2:
            pickle_output_file = os.path.join(nucleotide_folder, "dna_profiles_process_{}.pickle".format(i))
            save_mrc_folder = os.path.join(nucleotide_folder, "maps","dna")

            pdb_paths_process = dna_pdb_paths
            
            process = multiprocessing.Process(target=compute_and_save_radial_profiles, 
                                              args=(pdb_paths_process,  save_mrc_folder, pickle_output_file,i))
            
            jobs.append(process)
        elif i == 3:
            pickle_output_file = os.path.join(nucleotide_folder, "rna_profiles_process_{}.pickle".format(i))
            save_mrc_folder = os.path.join(nucleotide_folder, "maps","rna")

            pdb_paths_process = rna_pdb_paths
            
            process = multiprocessing.Process(target=compute_and_save_radial_profiles, 
                                              args=(pdb_paths_process,  save_mrc_folder, pickle_output_file,i))
            
            jobs.append(process)
        else:
            continue
    
        
    for job in jobs:
        job.start()
    
    for job in jobs:
        job.join()
    
    print("Radial Process extraction complete!")
            

    
    
    
 
