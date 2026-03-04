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
# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
num_iterations = 10

def copy_files_to_folder(file, folder):
    import shutil
    # if copied file already exists then ignore 
    test_copied_path = os.path.join(folder, os.path.basename(file))
    if os.path.exists(test_copied_path):
        print("File already exists: {}".format(test_copied_path))
        return test_copied_path
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    new_path = shutil.copy(file, folder)
    return new_path

def replace_atoms_with_pseudo_atoms(atomic_model_path):
    st = gemmi.read_structure(atomic_model_path)
    
    for cra in st[0].all():
        cra.atom.name = "O"
        cra.atom.element = gemmi.Element("O")
        cra.residue.name = "HOH"
    return st

def run_perturbation_analysis(input_unsharpened_map, input_refined_pdb, perturbed_refinement_folder_main):
    from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask
    from locscale.include.emmer.ndimage.map_utils import load_map
    from locscale.preprocessing.headers import run_servalcat_iterative
    from locscale.utils.map_quality import map_quality_pdb
    from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask

    emmap_basename = os.path.basename(input_unsharpened_map).split(".")[0]
    JOB_ID = emmap_basename
    perturbed_refinement_folder = os.path.join(perturbed_refinement_folder_main, JOB_ID)
    if not os.path.exists(perturbed_refinement_folder):
        os.makedirs(perturbed_refinement_folder)

    emmap_path_local = copy_files_to_folder(input_unsharpened_map, perturbed_refinement_folder)
    refined_pdb_path_local = copy_files_to_folder(input_refined_pdb, perturbed_refinement_folder)
    atomic_model_to_pseudomodel = replace_atoms_with_pseudo_atoms(refined_pdb_path_local)
    atomic_model_to_pseudomodel_path = os.path.join(perturbed_refinement_folder, "atomic_model_to_pseudomodel.cif")
    atomic_model_to_pseudomodel.make_mmcif_document().write_file(atomic_model_to_pseudomodel_path)

    # Get atomic model mask
    
    atomic_model_mask_path = os.path.join(perturbed_refinement_folder, "atomic_model_mask.mrc")
    atomic_model_mask_path = get_atomic_model_mask(emmap_path=emmap_path_local, pdb_path=refined_pdb_path_local, \
                                                output_filename = atomic_model_mask_path, save_files=True)

    # %%
    
    refinement_paths_input = {}
    emmap, apix = load_map(emmap_path_local)
    perturbation_magnitude = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    # Measure map to model correlation
    #%% 
    

    metric = map_quality_pdb(emmap_path_local, atomic_model_mask_path, refined_pdb_path_local)
    print(metric)
    assert metric > 0.2, "Map to model correlation is too low: {}".format(metric)

    for perturbation in perturbation_magnitude:
        perturbation_folder = os.path.join(perturbed_refinement_folder, "perturbation_{}_pm".format(int(perturbation*10)))
        if not os.path.exists(perturbation_folder):
            os.makedirs(perturbation_folder)
        perturbed_pdb_path = os.path.join(perturbation_folder, "perturbed_rmsd_{}_pm.cif".format(int(perturbation*10)))
        perturbed_pdb = shake_pdb_within_mask(pdb_path = atomic_model_to_pseudomodel, mask_path = atomic_model_mask_path, \
                                            rmsd_magnitude = perturbation, use_pdb_mask = False)
        perturbed_pdb.make_mmcif_document().write_file(perturbed_pdb_path)
        copied_emmap_path = copy_files_to_folder(emmap_path_local, perturbation_folder)

        refinement_paths_input[perturbation] = {
            "model_path": perturbed_pdb_path,
            "map_path": copied_emmap_path,
            "resolution": round(apix*2),
            "num_iter": num_iterations, 
            "pseudomodel_refinement" : True, 
        }

    # Atomic model refinement
    atomic_model_directory = os.path.join(perturbed_refinement_folder, "atomic_model_refinement")
    if not os.path.exists(atomic_model_directory):
        os.makedirs(atomic_model_directory)

    copied_emmap_path = copy_files_to_folder(emmap_path_local, atomic_model_directory)
    copied_pdb_path = copy_files_to_folder(refined_pdb_path_local, atomic_model_directory)

    refinement_paths_input["atomic_model"] = {
        "model_path": copied_pdb_path,
        "map_path": copied_emmap_path,
        "resolution": round(apix*2),
        "num_iter": num_iterations,
        "pseudomodel_refinement" : False,
    }


    # %%
    from joblib import Parallel, delayed

    refined_results = Parallel(n_jobs=9)(delayed(run_servalcat_iterative)(
        model_path = refinement_paths_input[key]["model_path"],            
        map_path = refinement_paths_input[key]["map_path"],
        resolution = refinement_paths_input[key]["resolution"],
        num_iter = refinement_paths_input[key]["num_iter"],
        pseudomodel_refinement = refinement_paths_input[key]["pseudomodel_refinement"],
    )
    for key in list(refinement_paths_input.keys()))

    return perturbed_refinement_folder

import gemmi
def get_coordinates_avg_bfactors_pdb(st, atomic_positions, window_size, ns):
    coordinates_bfactors_dict = {}
    for atom_pos in atomic_positions: 
        gemmi_pos = gemmi.Position(atom_pos[0], atom_pos[1], atom_pos[2])
        neighbors = ns.find_atoms(gemmi_pos, min_dist=0.1, radius=window_size//2)
        ADP_neighbors = [n.to_cra(st[0]).atom.b_iso for n in neighbors]
        avg_bfactor = np.mean(ADP_neighbors)
        coordinates_bfactors_dict[tuple(atom_pos)] = avg_bfactor
    # coordinates_bfactors_dict = {}
    # for cra in st[0].all():
    #     atom = cra.atom
    #     pos = tuple(atom.pos.tolist())
    #     bfactor = atom.b_iso
    #     #coordinates_bfactors_dict[pos] = bfactor
    #     average_bfactor = ns.find_neighbors(atom, min_dist=0.1, max_dist=window_size)
    #     coordinates_bfactors_dict[pos] = np.mean([n.to_cra(st[0]).atom.b_iso for n in average_bfactor])
    return coordinates_bfactors_dict

def get_coordinates_1(st):
    coordinates = []
    for cra in st[0].all():
        coordinates.append(cra.atom.pos.tolist())
    return coordinates

def get_coordinates_bfactors_pdb(st):
    coordinates_bfactors_dict = {}
    for cra in st[0].all():
        atom = cra.atom
        pos = tuple(atom.pos.tolist())
        bfactor = atom.b_iso
        coordinates_bfactors_dict[pos] = bfactor
    return coordinates_bfactors_dict

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "perturbation_study")

    high_resolution_emdb_path = os.path.join(data_input_folder_main, "experimental", "emd_0776_unsharpened.mrc")
    high_resolution_pdb_path = os.path.join(data_input_folder_main, "experimental", "6ku9.pdb")
    low_resolution_emdb_path = os.path.join(data_input_folder_main, "experimental", "emd_0492_unsharpened.mrc")
    low_resolution_pdb_path = os.path.join(data_input_folder_main, "experimental", "6nra.pdb")
    
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(high_resolution_emdb_path, high_resolution_pdb_path, low_resolution_emdb_path, low_resolution_pdb_path)
    # create_folders_if_they_do_not_exist(...) # for output folders
    output_filename = os.path.join(data_input_folder_main, "perturbation_analysis_results.pickle")
    # Run perturbation analysis for low resolution
    parent_folder_low_res_results = run_perturbation_analysis(low_resolution_emdb_path, low_resolution_pdb_path, data_input_folder_main)
    
    # Run perturbation analysis for high resolution
    parent_folder_high_res_results = run_perturbation_analysis(high_resolution_emdb_path, high_resolution_pdb_path, data_input_folder_main)


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

    for count, i in enumerate(int_perturbations):
        perturbed_structure_folder_high_res = os.path.join(parent_folder_high_res_results, "perturbation_{}_pm".format(i))
        perturbed_structure_folder_low_res = os.path.join(parent_folder_low_res_results, "perturbation_{}_pm".format(i))

        perturbed_refined_path_high_res = os.path.join(perturbed_structure_folder_high_res, f"servalcat_refinement_cycle_10.pdb")
        perturbed_refined_path_low_res = os.path.join(perturbed_structure_folder_low_res, f"servalcat_refinement_cycle_10.pdb")

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

