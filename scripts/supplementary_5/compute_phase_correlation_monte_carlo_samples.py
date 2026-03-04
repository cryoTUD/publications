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
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
from locscale.include.emmer.ndimage.fsc_util import calculate_phase_correlation_maps, calculate_amplitude_correlation_maps
from locscale.include.emmer.ndimage.profile_tools import frequency_array
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
    

    output_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "monte_carlo_phase_correlation")
    correlation_plot = os.path.join(output_folder_main, "amplitude_phase_correlation_monte_carlo.pdf")
    output_data_filename = os.path.join(output_folder_main, f"phase_correlation_monte_carlo_samples_{num_samples}.pickle")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(emmernet_model_path, cubes_input_folder)  # check if the paths exist
    create_folders_if_they_do_not_exist(output_folder_main)  # create output folders if they do not exist
    
    ## Do your processing here
    monte_carlo_map_paths = [os.path.join(cubes_input_folder, f"monte_carlo_sample_{i}.mrc") for i in range(num_samples)]
    # Load the cubes
    monte_carlo_cubes = np.zeros((len(monte_carlo_map_paths), 32, 32, 32))
    first_monte_carlo_map_path = monte_carlo_map_paths[0]
    phase_correlations_list = []
    amplitude_correlations_list = []
    for i, path in enumerate(monte_carlo_map_paths[1:]):
        # Load the cube
        cube, apix = load_map(path)
        phase_correlation = calculate_phase_correlation_maps(cube, first_monte_carlo_map_path)
        phase_correlations_list.append(phase_correlation)
        amplitude_correlation = calculate_amplitude_correlation_maps(cube, first_monte_carlo_map_path)
        amplitude_correlations_list.append(amplitude_correlation)
    
    freq = frequency_array(phase_correlations_list[0], apix=apix)

    figsize_mm = (70, 35)
    fontsize = 6
    rcparams = configure_plot_scaling(figsize_mm, fontsize)
    with temporary_rcparams(rcparams):
        fig, ax = plt.subplots(1, 2)
        for i, amplitude_correlation in enumerate(amplitude_correlations_list):
            ax[0].plot(freq, amplitude_correlation, color="black", alpha=0.1)
        ax[0].set_xlabel(r"Spatial Frequency ($\AA^{-1}$)")
        ax[0].set_ylabel("Amplitude Correlation")
        ax[0].set_ylim(0, 1.2)    

        for i, phase_correlation in enumerate(phase_correlations_list):
            ax[1].plot(freq, phase_correlation, color="black", alpha=0.1)
        
        ax[1].set_xlabel(r"Spatial Frequency ($\AA^{-1}$)")
        ax[1].set_ylabel("Phase Correlation")
        ax[1].set_ylim(0, 1.2)

        #fig.suptitle(f"Phase and Amplitude Correlation Monte Carlo Samples ({num_samples} samples)")

        plt.tight_layout()
        plt.savefig(correlation_plot)

    # Save the data
    with open(output_data_filename, "wb") as f:
        pickle.dump({"phase_correlations": phase_correlations_list, "amplitude_correlations": amplitude_correlations_list}, f)
    print(f"Saved files to {output_folder_main}")
    print(f"Saved correlation_plot to {correlation_plot}")
    print(f"Saved correlation data to {output_data_filename}")


    
if __name__ == "__main__":
    main()

