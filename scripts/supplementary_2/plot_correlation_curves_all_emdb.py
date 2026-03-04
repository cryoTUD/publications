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

        correlations_restrained_pseudomodels_all[emdb_pdb] = correlations_restrained
        correlations_unrestrained_pseudomodels_all[emdb_pdb] = correlations_unrestrained
        correlations_atomic_model_all[emdb_pdb] = correlations_atomic_model

    # Plotting
    xdata = list(range(1,11))
    ydata_restrained = [x for x in list(correlations_restrained_pseudomodels_all.values()) if np.all(~np.isnan(x))]
    ydata_unrestrained = [x for x in list(correlations_unrestrained_pseudomodels_all.values()) if np.all(~np.isnan(x))]
    ydata_atomic_model = [ x for x in list(correlations_atomic_model_all.values()) if np.all(~np.isnan(x))]

    print(f"Number of EMDB-PDB pairs: {len(correlations_restrained_pseudomodels_all)}")
    figsize_mm = (80, 80) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
        fontsize = 8
        ylim = (0.2,1.2)
        yticks = [0.5,1.0]
        fig = pretty_lineplot_XY_multiple_with_shade(\
                xdata, [ydata_atomic_model, ydata_restrained, ydata_unrestrained],  
                xlabel=r"Neighborhood Radius ($\AA$)", ylabel="ADP Correlation",
                figsize_cm=figsize_cm, fontsize=fontsize, ylims=ylim, yticks=yticks,
                save_path=plot_output_filename_correlation_curve,
        )

    print(f"Plot saved to {plot_output_filename_correlation_curve}. Please check.")

if __name__ == "__main__":
    main()

