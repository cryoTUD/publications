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

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, plot_binned_residuals
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.emmernet.emmernet_functions import load_smoothened_mask

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "maps")
    feature_enhance_folder = os.path.join(data_input_folder_main, "feature_enhance_test_maps_hybrid_60k")
    confidence_mask_folder = os.path.join(data_input_folder_main, "confidence_masks")
    #target_maps_folder_1 = "/home/abharadwaj1/papers/elife_paper/figure_information/data/hybrid_model_maps_version_C"
    target_maps_folder_main = "/tudelft/abharadwaj1/staff-umbrella/ajlab/AB/PhD_research/cryo_em_map_sharpening/papers_and_conference/elife_paper/figure_information"
    target_maps_folder = os.path.join(target_maps_folder_main, "data", "hybrid_model_maps_version_C")
    # figure_input_folder = /add/your/path/here
    # other input folder 

    output_folder = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "residuals_analysis")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(feature_enhance_folder, confidence_mask_folder)
    create_folders_if_they_do_not_exist(output_folder)
    
    output_filename = os.path.join(output_folder, "residuals_analysis_redo.pickle")
    emdb_pdbs = [x for x in os.listdir(feature_enhance_folder)]

    input_files_emdb = {}
    for emdb_pdb in emdb_pdbs:
        emdb, pdb = emdb_pdb.split("_")
        
        emdb_prediction_folder = os.path.join(feature_enhance_folder, emdb_pdb)
        mean_prediction_path = os.path.join(emdb_prediction_folder, f"emd_{emdb}_emmernet_output_mean.mrc")
        var_prediction_path = os.path.join(emdb_prediction_folder, f"emd_{emdb}_emmernet_output_var.mrc")
        mask_path = os.path.join(confidence_mask_folder, f"emd_{emdb}_FDR_confidence_final.map")
        locscale_map_path = os.path.join(emdb_prediction_folder, f"emd_{emdb}_emmernet_output_locscale_output.mrc")
        target_map_path = os.path.join(target_maps_folder, f"emd_{emdb}_hybrid_model_map_refined_version_C.mrc")
        if not os.path.exists(target_map_path):
            print(f"Target map not found for {emdb_pdb}, skipping...")
            continue

        input_files_emdb[emdb_pdb] = {
            "mean_prediction_path": mean_prediction_path,
            "var_prediction_path": var_prediction_path,
            "mask_path": mask_path,
            "locscale_map_path": locscale_map_path,
            "target_map_path": target_map_path
        }

    # assert all paths exist
    for emdb_pdb, paths in input_files_emdb.items():
        assert_paths_exist(*paths.values())
    
    # Process the data
    residuals_emdb = {}
    squared_residuals_emdb = {}
    masked_variance_emdb = {}
    for emdb_pdb, paths in tqdm(input_files_emdb.items()):
        mean_prediction_path = paths["mean_prediction_path"]
        var_prediction_path = paths["var_prediction_path"]
        mask_path = paths["mask_path"]
        locscale_map_path = paths["locscale_map_path"]
        target_map_path = paths["target_map_path"]

        # Load the data
        mean_prediction = load_map(mean_prediction_path)[0]
        var_prediction = load_map(var_prediction_path)[0]
        mask, _ = load_smoothened_mask(mask_path, mask_threshold=0.99)
        locscale_map, apix = load_map(locscale_map_path)
        target_map, _ = load_map(target_map_path)

        mask_binarised = mask > 0.5
        masked_prediction = mean_prediction[mask_binarised]
        masked_var_prediction = var_prediction[mask_binarised]
        masked_locscale_map = locscale_map[mask_binarised]
        masked_target_map = target_map[mask_binarised]

        # Calculate residuals
        residuals = masked_prediction - masked_locscale_map
        squared_residuals = residuals ** 2
        # Store the results
        residuals_emdb[emdb_pdb] = residuals
        squared_residuals_emdb[emdb_pdb] = squared_residuals
        masked_variance_emdb[emdb_pdb] = masked_var_prediction
    
    max_residuals = max([max(residuals) for residuals in residuals_emdb.values()])
    print(f"Max residual across all EMDB entries: {max_residuals}")
    # Save the results
    # with open(output_filename, "wb") as f:
    #     pickle.dump({
    #         "residuals_emdb": residuals_emdb,
    #         "squared_residuals_emdb": squared_residuals_emdb,
    #         "masked_variance_emdb": masked_variance_emdb
    #     }, f) 

    # print(f"Data saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

