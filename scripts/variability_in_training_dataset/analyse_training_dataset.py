## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pickle
import pandas as pd
# Custom imports
from utils.chapter_3_functions import get_cumulative_probability_threshold_levels, get_2d_jointplot_with_text
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams
from tqdm import tqdm

# Set the seed for reproducibility
np.random.seed(42)

## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    training_data_features_path = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "training_data_analysis", "training_targets_feature_info.pickle")
    output_folder = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "training_data_analysis")
    assert_paths_exist(training_data_features_path)
    
    output_filename = os.path.join(output_folder, "flatness_fragmentation_ratios.pickle")
    
    # Load the training data features
    with open(training_data_features_path, 'rb') as f:
        training_data_features = pickle.load(f)
    
    print(f"There are {len(training_data_features)} entries in the training data. Expected 170.")

    # Extract flatness and fragmentation ratios
    flatness_fragmentation_ratios = {} 
    for emdb_id in tqdm(training_data_features.keys(), desc="Processing training data"):
        length_of_micelle = training_data_features[emdb_id]["features"]["length"]
        height_of_micelle = training_data_features[emdb_id]["features"]["height"]
        # Extract the information from the training data
        fragmentation_ratio = training_data_features[emdb_id]["features"]["surface_volume_ratio"]
        flatness_ratio = 1 - height_of_micelle / length_of_micelle
        flatness_fragmentation_ratios[emdb_id] = {
            "flatness": flatness_ratio, 
            "fragmentation": fragmentation_ratio,
            "length": length_of_micelle,
            "height": height_of_micelle
        }

    all_flatness_ratios = [flatness_fragmentation_ratios[emdb_id]["flatness"] for emdb_id in flatness_fragmentation_ratios.keys()]  
    all_fragmentation_ratios = [flatness_fragmentation_ratios[emdb_id]["fragmentation"] for emdb_id in flatness_fragmentation_ratios.keys()]
    print(f"Flatness: {np.mean(all_flatness_ratios):.2f} +/- {np.std(all_flatness_ratios):.2f}")
    print(f"Fragmentation: {np.mean(all_fragmentation_ratios):.2f} +/- {np.std(all_fragmentation_ratios):.2f}")

    flatness_fragmentation_ratios['all_flatness_ratios'] = all_flatness_ratios
    flatness_fragmentation_ratios['all_fragmentation_ratios'] = all_fragmentation_ratios

    # Save the flatness and fragmentation ratios

    pd.to_pickle(flatness_fragmentation_ratios, output_filename)

if __name__ == "__main__":
    main()
