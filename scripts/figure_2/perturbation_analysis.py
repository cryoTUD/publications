## IMPORTS
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import json
import pandas as pd
import random 
import gemmi
from tqdm import tqdm
# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from locscale.include.emmer.ndimage.map_tools import estimate_global_bfactor_map_standard
from figure_2_functions import run_perturbation_analysis, get_coordinates_bfactors_pdb
# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
num_iterations = 10
run_analysis = True

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "perturbation_study")

    high_resolution_emdb_path = os.path.join(data_input_folder_main, "experimental", "EMD_20521_unsharpened_fullmap.mrc")
    high_resolution_pdb_path = os.path.join(data_input_folder_main, "experimental", "PDB_6pxm_unrefined_shifted_servalcat_refined_20521.pdb")
    low_resolution_emdb_path = os.path.join(data_input_folder_main, "experimental", "EMD_4141_unsharpened_fullmap.mrc")
    low_resolution_pdb_path = os.path.join(data_input_folder_main, "experimental", "PDB_5m1s_unrefined_shifted_servalcat_refined_4141.pdb")
    
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(high_resolution_emdb_path, high_resolution_pdb_path, low_resolution_emdb_path, low_resolution_pdb_path)
    # create_folders_if_they_do_not_exist(...) # for output folders
    output_filename = os.path.join(data_input_folder_main, "perturbation_analysis_results.pickle")
    if run_analysis:
        # Run perturbation analysis for low resolution
        parent_folder_low_res_results = run_perturbation_analysis(\
            low_resolution_emdb_path, low_resolution_pdb_path, data_input_folder_main, num_iterations=num_iterations)
        
        # Run perturbation analysis for high resolution
        parent_folder_high_res_results = run_perturbation_analysis(\
            high_resolution_emdb_path, high_resolution_pdb_path, data_input_folder_main, num_iterations=num_iterations)
    else:
        emmap_basename_low_res = os.path.basename(low_resolution_emdb_path).split(".")[0]
        emmap_basename_high_res = os.path.basename(high_resolution_emdb_path).split(".")[0]
        parent_folder_low_res_results = os.path.join(data_input_folder_main, emmap_basename_low_res)
        parent_folder_high_res_results = os.path.join(data_input_folder_main, emmap_basename_high_res)
        


    atomic_model_results_high_res = os.path.join(parent_folder_high_res_results, "atomic_model_refinement")
    #pseudo_atomic_model_results_high_res = os.path.join(parent_folder_high_res_results, "using_pseudo_atomic_model")

    atomic_model_results_low_res = os.path.join(parent_folder_low_res_results, "atomic_model_refinement")
    #pseudo_atomic_model_results_low_res = os.path.join(parent_folder_low_res_results, "pseudomodel_refinement")

    high_resolution_pdb_filename = os.path.basename(high_resolution_pdb_path)
    low_resolution_pdb_filename = os.path.basename(low_resolution_pdb_path)

    refined_atomic_model_path_high_res = os.path.join(atomic_model_results_high_res, high_resolution_pdb_filename.replace(".pdb", "_servalcat_refined.pdb"))
    #refined_pseudo_atomic_model_path_high_res = os.path.join(pseudo_atomic_model_results_high_res, "emd_20521_FDR_confidence_final_gradient_pseudomodel_uniform_biso_proper_element_composition.pdb")

    refined_atomic_model_path_low_res = os.path.join(atomic_model_results_low_res, low_resolution_pdb_filename.replace(".pdb", "_servalcat_refined.pdb"))
    #refined_pseudo_atomic_model_path_low_res = os.path.join(pseudo_atomic_model_results_low_res, "emd_4141_FDR_confidence_final_gradient_pseudomodel_proper_element_composition_proper_element_composition.pdb")
    assert os.path.exists(refined_atomic_model_path_high_res), f"File {refined_atomic_model_path_high_res} does not exist"
    assert os.path.exists(refined_atomic_model_path_low_res), f"File {refined_atomic_model_path_low_res} does not exist"

    

    bfactor_low_res = estimate_global_bfactor_map_standard(low_resolution_emdb_path, wilson_cutoff=10, fsc_cutoff=6.7)
    bfactor_high_res = estimate_global_bfactor_map_standard(high_resolution_emdb_path, wilson_cutoff=10, fsc_cutoff=2.1)

    print("Low resolution B-factor: ", bfactor_low_res)
    print("High resolution B-factor: ", bfactor_high_res)

    perturbation_magnitudes = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    int_perturbations = [int(i*10) for i in perturbation_magnitudes]

    perturbed_structures_high_res = {}
    perturbed_structures_low_res = {}

    for count, perturb_mag in enumerate(int_perturbations):
        perturbed_structure_folder_high_res = os.path.join(parent_folder_high_res_results, "perturbation_{}_pm".format(perturb_mag))
        perturbed_structure_folder_low_res = os.path.join(parent_folder_low_res_results, "perturbation_{}_pm".format(perturb_mag))

        perturbed_refined_path_high_res = os.path.join(perturbed_structure_folder_high_res, \
                                                        f"perturbed_rmsd_{perturb_mag}_pm_proper_element_composition_averaged.cif")
        perturbed_refined_path_low_res = os.path.join(perturbed_structure_folder_low_res, \
                                                        f"perturbed_rmsd_{perturb_mag}_pm_proper_element_composition_averaged.cif")
        

        assert os.path.exists(perturbed_refined_path_high_res), "Path does not exist: {}".format(perturbed_refined_path_high_res)
        assert os.path.exists(perturbed_refined_path_low_res), "Path does not exist: {}".format(perturbed_refined_path_low_res)
        perturb_rmsd = perturbation_magnitudes[count]
        perturbed_structures_high_res[perturb_rmsd] = perturbed_refined_path_high_res
        perturbed_structures_low_res[perturb_rmsd] = perturbed_refined_path_low_res
    
    
    perturbed_average_bfactors_high_res = {}
    perturbed_average_bfactors_low_res = {}

    # Combine perturbed structures , atomic model and pseudo-atomic model into one dictionary for each resolution
    perturbed_structures_high_res["atomic_model"] = refined_atomic_model_path_high_res
    #perturbed_structures_high_res["pseudo_atomic_model"] = refined_pseudo_atomic_model_path_high_res

    perturbed_structures_low_res["atomic_model"] = refined_atomic_model_path_low_res
    #perturbed_structures_low_res["pseudo_atomic_model"] = refined_pseudo_atomic_model_path_low_res

    # Check if all paths exist
    for key, pdb_path in perturbed_structures_high_res.items():
        assert os.path.exists(pdb_path), f"{pdb_path} does not exist"

    for key, pdb_path in perturbed_structures_low_res.items():
        assert os.path.exists(pdb_path), f"{pdb_path} does not exist"
        
    # Get average Bfactors for each perturbed structure
    for key, pdb_path in tqdm(perturbed_structures_high_res.items()):
        st = gemmi.read_structure(pdb_path)
        ns = gemmi.NeighborSearch(st[0], st.cell, 25).populate()
        perturbed_average_bfactors_high_res[key] = list(get_coordinates_bfactors_pdb(st).values())

    for key, pdb_path in tqdm(perturbed_structures_low_res.items()):
        st = gemmi.read_structure(pdb_path)
        ns = gemmi.NeighborSearch(st[0], st.cell, 25).populate()
        perturbed_average_bfactors_low_res[key] = list(get_coordinates_bfactors_pdb(st).values())
    

    # Save the results
    output_dictionary = {
        "high_resolution_emdb_path": high_resolution_emdb_path,
        "high_resolution_pdb_path": high_resolution_pdb_path,
        "low_resolution_emdb_path": low_resolution_emdb_path,
        "low_resolution_pdb_path": low_resolution_pdb_path,
        "perturbed_average_bfactors_high_res" : perturbed_average_bfactors_high_res,
        "perturbed_average_bfactors_low_res" : perturbed_average_bfactors_low_res,
        "perturbation_magnitudes" : perturbation_magnitudes,
        "bfactor_low_res" : bfactor_low_res,
        "bfactor_high_res" : bfactor_high_res,
        
    }

    with open(output_filename, "wb") as f:
        pickle.dump(output_dictionary, f)

    print("Perturbation analysis completed.")
    

if __name__ == "__main__":
    main()

