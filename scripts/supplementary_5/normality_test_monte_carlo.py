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
from scipy.stats import shapiro
# anderson darling test 
from scipy.stats import anderson

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)
num_samples = 50
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "general", "supplementary_5")
    emmernet_model_path = os.path.join(data_input_folder_main, "EMmerNet_highContext.hdf5")
    cubes_input_folder = os.path.join(data_input_folder_main, "8069_cube_77")
    # figure_input_folder = /add/your/path/here
    

    output_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "normality_test")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(emmernet_model_path, cubes_input_folder)  # check if the paths exist
    create_folders_if_they_do_not_exist(output_folder_main)  # create output folders if they do not exist
    
    output_filename_mean_map = os.path.join(output_folder_main, f"mean_map_num_samples_{num_samples}.mrc")
    output_filename_p_values_map = os.path.join(output_folder_main, f"p_values_map_num_samples_{num_samples}.mrc")
    output_filename_significant_deviations_map = os.path.join(output_folder_main, f"significant_deviations_map_num_samples_{num_samples}.mrc")

    ## Do your processing here
    monte_carlo_map_paths = [os.path.join(cubes_input_folder, f"monte_carlo_sample_{i}.mrc") for i in range(num_samples)]
    # Load the cubes
    monte_carlo_cubes = np.zeros((len(monte_carlo_map_paths), 32, 32, 32))
    for i, path in enumerate(monte_carlo_map_paths):
        # Load the cube
        cube, apix = load_map(path)
        monte_carlo_cubes[i] = cube

    p_values_map = np.zeros((32, 32, 32))
    significant_deviations_map = np.zeros((32, 32, 32))
    standardize = lambda x: (x - np.mean(x)) / np.std(x)

    # compute p values along the values along the 0th axis
    for i in tqdm(range(32)):
        for j in range(32):
            for k in range(32):
                # Perform the Shapiro-Wilk test
                stat, p_value = shapiro(standardize(monte_carlo_cubes[:, i, j, k]))
                p_values_map[i, j, k] = p_value
                # Check if the p-value is less than 0.05
                if p_value < 0.05:
                    significant_deviations_map[i, j, k] = 1
    
    mean_map = np.mean(monte_carlo_cubes, axis=0)
    
    # Save the mean map, p-values map, and significant deviations map
    save_as_mrc(mean_map, output_filename_mean_map, apix)
    save_as_mrc(p_values_map, output_filename_p_values_map, apix)
    save_as_mrc(significant_deviations_map, output_filename_significant_deviations_map, apix)

    print(f"Saved files to {output_folder_main}")

if __name__ == "__main__":
    main()

