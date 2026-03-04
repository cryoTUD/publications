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

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "inputs")
    input_file_path = os.path.join(data_input_folder_main, "", "", "XXXXXX.json")
    # figure_input_folder = /add/your/path/here
    # other input folder 

    output_folder_main = os.path.join(data_archive_path, "processed_data_output")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(...)
    create_folders_if_they_do_not_exist(...) # for output folders
    
    output_filename = os.path.join(output_folder_main, "XXXXXX.json")  # output plot preferably in json

    # Load the training data features
    with open(input_file_path, 'r') as f:
        input_dictionary = json.load(f)

    ## Do your processing here
        
    output_data = [] 
    # Save the data
    with open(output_filename, 'w') as f:
        json.dump(output_data, f)

    print(f"Data saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

