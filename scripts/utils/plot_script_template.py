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
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "inputs")
    # figure_input_folder = /add/your/path/here
    # other input folder 

    figure_output_folder_main = os.path.join(data_archive_path, "figures_output")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(...)
    create_folders_if_they_do_not_exist(...) # for output folders
    
    output_filename = os.path.join(plot_output_folder, "XXXXXX.pdf")  # output plot preferably in pdf format

    # Load the training data features
    # with open(pickle_file_path, 'rb') as f:
    #     input_dictionary = pickle.load(f)

    ## Do your processing here

    ## Plotting
        
    figsize_mm = (50, 30) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        # Plotting code here
        plt.plot(...) # or any other plotting function

    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

