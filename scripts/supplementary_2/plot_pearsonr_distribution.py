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
from scipy.stats import pearsonr

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, pretty_lineplot_XY_multiple_with_shade

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    output_folder_main = os.path.join(data_archive_path, "structured_data", "supplementary_2")
    create_folders_if_they_do_not_exist(output_folder_main)
    output_filename = os.path.join(output_folder_main, "ADP_correlation_curves_all_emdb.pickle")  # output plot preferably in json
    
    figure_output_folder_main = os.path.join(data_archive_path, "outputs","supplementary_2", "neighborhood_correlation", "many_maps")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(output_filename)
    create_folders_if_they_do_not_exist(figure_output_folder_main) # for output folders
    
    plot_output_filename_correlation_curve = os.path.join(figure_output_folder_main, "correlation_curves_all.pdf")  # output plot preferably in pdf format
    plot_output_filename_pearsonr = os.path.join(figure_output_folder_main, "pearsonr_distribution_all.pdf")  # output plot preferably in pdf format

    # Load the training data features
    with open(output_filename, 'rb') as f:
        input_dictionary = pickle.load(f)

    ## Do your processing here
    adp_correlation_curves_pseudomodel_restrained = input_dictionary["adp_correlation_curves_pseudomodel_restrained"]
    adp_correlation_curves_pseudomodel_unrestrained = input_dictionary["adp_correlation_curves_pseudomodel_unrestrained"]
    adp_correlation_curves_atomic_model = input_dictionary["adp_correlation_curves_atomic_model"]
    EMDB_PDB_ids_present = input_dictionary["EMDB_PDB_ids_present"]

    correlations_restrained_pseudomodels_all = {}
    correlations_unrestrained_pseudomodels_all = {}
    correlations_atomic_model_all = {}

    pearson_correlations_restrained_pseudomodels_all = {}
    pearson_correlations_unrestrained_pseudomodels_all = {}
    for emdb_pdb in EMDB_PDB_ids_present:
        if emdb_pdb in ["0026_6gl7"]:
            continue
        emdb, pdb = emdb_pdb.split("_")

        neighborhood_correlation_curves_restrained = adp_correlation_curves_pseudomodel_restrained[emdb_pdb]
        neighborhood_correlation_curves_unrestrained = adp_correlation_curves_pseudomodel_unrestrained[emdb_pdb]
        neighborhood_correlation_curves_atomic_model = adp_correlation_curves_atomic_model[emdb_pdb]

        correlations_restrained = [x[2][0] for x in neighborhood_correlation_curves_restrained.values()]
        correlations_unrestrained = [x[2][0] for x in neighborhood_correlation_curves_unrestrained.values()]
        correlations_atomic_model = [x[2][0] for x in neighborhood_correlation_curves_atomic_model.values()]

        # Calculate Pearson correlations between atomic model and restrained/unrestrained pseudomodels
        if np.any(np.isnan(correlations_restrained)) or np.any(np.isnan(correlations_unrestrained)) or np.any(np.isnan(correlations_atomic_model)):
            print(f"Skipping {emdb_pdb} due to NaN values in correlations.")
            continue
        r_value_restrained, p_value_restrained = pearsonr(correlations_restrained, correlations_atomic_model)
        r_value_unrestrained, p_value_unrestrained = pearsonr(correlations_unrestrained, correlations_atomic_model)

        pearson_correlations_restrained_pseudomodels_all[emdb_pdb] = r_value_restrained
        pearson_correlations_unrestrained_pseudomodels_all[emdb_pdb] = r_value_unrestrained

    # Plotting
    pearson_values_restrained = [x for x in list(pearson_correlations_restrained_pseudomodels_all.values()) if not np.isnan(x)]
    pearson_values_unrestrained = [x for x in list(pearson_correlations_unrestrained_pseudomodels_all.values()) if not np.isnan(x)]

    figsize_mm = (60, 80) # width, height
    fontsize = 8
    rcparams = configure_plot_scaling(figsize_mm)
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(1, 1, figsize=figsize_in, dpi=600)
        ax.violinplot([pearson_values_unrestrained, pearson_values_restrained], showmeans=False, showmedians=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Unrestrained refinement", "Restrained refinement"], rotation=45, ha="right")
        ax.set_ylabel("Pearson correlation coefficient")
        fig.tight_layout()

        plt.savefig(plot_output_filename_pearsonr, dpi=600)


    print(f"Plot saved to {plot_output_filename_pearsonr}. Please check.")

if __name__ == "__main__":
    main()

