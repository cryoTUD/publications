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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, plot_correlations

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    json_data_path = os.path.join(data_archive_path, "outputs", "supplementary_2", "neighborhood_correlation", "neighborhood_correlation_data.json")
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2", "neighborhood_correlation")

    assert_paths_exist(json_data_path)
    create_folders_if_they_do_not_exist(plot_output_folder)

    # Load data
    with open(json_data_path, 'r') as f:
        data = json.load(f)

    x_label = r"ADP ($\AA^2$)"
    y_label = r"$\langle$ ADP $\rangle$ ($\AA^2$)"

    figsize_mm = (80, 80)
    fontsize = 8
    rcparams = configure_plot_scaling(figsize_mm)

    for radius in [2, 10]:
        key = f"radius_{radius}"
        title = f"r={radius} $\AA$"
        for model_key, label in zip(
            ["atomic", "restrained_pseudomodel", "unrestrained_pseudomodel"],
            ["atomic", "restrained", "unrestrained"]):

            x = np.array(data[key][model_key]["individual_bfactors"])
            y = np.array(data[key][model_key]["neighborhood_bfactors"])

            output_filename = os.path.join(
                plot_output_folder, f"correlation_{label}_r{radius}.pdf")

            with temporary_rcparams(rcparams):
                fig = plot_correlations(
                    x_array=x, y_array=y,
                    x_label=x_label, y_label=y_label,
                    figsize_cm=(figsize_mm[0] / 10, figsize_mm[1] / 10),
                    fontsize=fontsize, fontscale=1, font="Arial", scatter=True,
                    xticks=[100, 200, 300], yticks=[100, 200, 300],
                    filepath=output_filename, 
                )
                

            print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()
