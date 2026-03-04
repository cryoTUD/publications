## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
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
    data_input_folder_main = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "perturbation_study")
    input_filename = os.path.join(data_input_folder_main, "perturbation_analysis_results.pickle")

    figure_output_folder_main = os.path.join(data_archive_path, "figures_output", "figure_2")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(input_filename)
    create_folders_if_they_do_not_exist(figure_output_folder_main) # for output folders
    
    output_filename = os.path.join(figure_output_folder_main, "perturbation_analysis_experimental.pdf")  # output plot preferably in pdf format

    # Load the training data features
    with open(input_filename, 'rb') as f:
        input_dictionary = pickle.load(f)

    ## Do your processing here
    perturbed_average_bfactors_high_res = input_dictionary["perturbed_average_bfactors_high_res"]
    perturbed_average_bfactors_low_res = input_dictionary["perturbed_average_bfactors_low_res"]
    bfactor_high_res = input_dictionary["bfactor_high_res"]
    bfactor_low_res = input_dictionary["bfactor_low_res"]
    high_resolution_emdb_path = input_dictionary["high_resolution_emdb_path"]
    low_resolution_emdb_path = input_dictionary["low_resolution_emdb_path"]
    high_resolution_emdb = os.path.basename(high_resolution_emdb_path).split("_")[1]
    low_resolution_emdb = os.path.basename(low_resolution_emdb_path).split("_")[1]
    ## Plotting
        
    figsize_mm = (50, 60) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        # Plotting code here
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_in, sharex=True, sharey=True, dpi=600)
    
        xticklabels = list(perturbed_average_bfactors_high_res.keys())[:-1]
        xtick_labels_string = [f"{x} $\AA$" for x in xticklabels]
        high_res_values = list(perturbed_average_bfactors_high_res.values())[:-1]
        low_res_values = list(perturbed_average_bfactors_low_res.values())[:-1]
        ax1.violinplot(high_res_values, showmeans=False, showmedians=True)
        #ax1.set_title("High Resolution {}".format(high_resolution_emmap_filename))
        ax1.set_ylabel(r" $\langle$ADP$\rangle$ $(\AA^2)$ ")
        ax1.set_xticks(range(1, len(perturbed_average_bfactors_high_res.keys())))
        #ax1.set_xticklabels(perturbed_average_bfactors_high_res.keys(), rotation=10)
        ax1.set_xticklabels(xtick_labels_string, rotation=10)
        ax1_text = f"EMDB: {high_resolution_emdb}\nFSC: 2.1 $\AA$\nB-factor: {bfactor_high_res:.2f} $\AA^2$"
        ax1.text(0.05, 0.95, ax1_text, transform=ax1.transAxes, fontsize=6, verticalalignment='top')

        ax2.violinplot(low_res_values, showmeans=False, showmedians=True)
        #ax2.set_title("Low Resolution {}".format(low_resolution_emdb_filename))
        ax2.set_ylabel(r" $\langle$ADP$\rangle$ $(\AA^2)$ ")
        ax2.set_xticks(range(1, len(perturbed_average_bfactors_low_res.keys())))
        #ax2.set_xticklabels(perturbed_average_bfactors_low_res.keys(), rotation=10)
        ax2.set_xticklabels(xtick_labels_string, rotation=45)
        ax2_text = f"EMDB: {low_resolution_emdb}\nFSC: 6.7 $\AA$\nB-factor: {bfactor_low_res:.2f} $\AA^2$"
        ax2.text(0.6, 0.9, ax2_text, transform=ax2.transAxes, fontsize=6, verticalalignment='top')

        fig.savefig(output_filename, bbox_inches='tight')


    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

