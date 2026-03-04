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
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
from locscale.include.emmer.ndimage.map_utils import load_map

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "monte_carlo_samples")
    list_of_means = os.path.join(data_input_folder_main, "list_of_means500.json")
    list_of_variances = os.path.join(data_input_folder_main, "list_of_variances500.json")
    
    # figure_input_folder = /add/your/path/here
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_5")
    plot_output_filename = os.path.join(plot_output_folder, "variance_evolution.pdf")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(list_of_means, list_of_variances)  # check if the paths exist
    create_folders_if_they_do_not_exist(plot_output_folder)  # create output folders if they do not exist
    
    list_of_means = json.load(open(list_of_means))
    list_of_variances = json.load(open(list_of_variances))
    
    print(f"Loaded {len(list_of_means)} means and {len(list_of_variances)} variances")
    num_samples_in_each_cube = len(list_of_means[0])
    print(f"Number of samples in each cube: {num_samples_in_each_cube}")
    list_of_means_new = [[float(x) for x in old_list] for old_list in list_of_means]
    list_of_variances_new = [[float(x) for x in old_list] for old_list in list_of_variances]
    figsize_mm = (60, 40)
    fontsize = 6
    rcparams = configure_plot_scaling(figsize_mm, fontsize)
    with temporary_rcparams(rcparams):
        fig, ax = plt.subplots(1, 1)
        for i in range(len(list_of_means_new)):    
            ax.plot(list_of_variances_new[i], color='black', alpha=0.025)
            #ax.set_title("Variances")
            ax.set_xlabel("Monte Carlo Samples")
            ax.set_ylabel("Variance")
            ax.set_yscale("log")
            ax.set_yticks([1e-9, 1e-7, 1e-5, 1e-3])
            ax.set_ylim(1e-10, 1e-2)
            ax.set_xscale("log")
            # draw vertical line at 15 
            ax.axvline(x=15, color='red', linestyle='--')
            #ax[1].set_ylim(0, 1)
            
        
    fig.tight_layout()
    # Save figure 
    fig.savefig(plot_output_filename, dpi=600)
    print(f"Figure saved to {plot_output_filename}")

if __name__ == "__main__":
    main()

