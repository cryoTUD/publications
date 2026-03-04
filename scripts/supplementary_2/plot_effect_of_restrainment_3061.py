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
    analysis_output_folder = os.path.join(data_archive_path, "processed", "structured_data", "supplementary_2")
    output_filename = os.path.join(analysis_output_folder, "fsc_curves_with_and_without_averaging.json")
    # plot_output_folder = /add/your/path/here
    # other output folder
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2")
    assert_paths_exist(output_filename)
    create_folders_if_they_do_not_exist(plot_output_folder) # for output folders
    
    plot_output_filename = os.path.join(plot_output_folder, "effect_of_restrainment_3061.pdf")  # output plot preferably in pdf format

    # Load the training data features
    with open(output_filename, 'r') as f:
        input_dictionary = json.load(f)

    ## Do your processing here
    fsc_cycles_halfmap1_without_averaging = input_dictionary["fsc_cycles_halfmap1_without_averaging"]
    fsc_cycles_halfmap2_without_averaging = input_dictionary["fsc_cycles_halfmap2_without_averaging"]
    fsc_cycles_halfmap1_with_averaging = input_dictionary["fsc_cycles_halfmap1_with_averaging"]
    fsc_cycles_halfmap2_with_averaging = input_dictionary["fsc_cycles_halfmap2_with_averaging"]
    refmac_iterations = input_dictionary["cycles"]

    # print type of data to check

    fsc_average_curve_halfmap1_without_averaging = [fsc_cycles_halfmap1_without_averaging[cycle][1] for cycle in refmac_iterations]
    fsc_average_curve_halfmap2_without_averaging = [fsc_cycles_halfmap2_without_averaging[cycle][1] for cycle in refmac_iterations]
    fsc_average_curve_halfmap1_with_averaging = [fsc_cycles_halfmap1_with_averaging[cycle][1] for cycle in refmac_iterations]
    fsc_average_curve_halfmap2_with_averaging = [fsc_cycles_halfmap2_with_averaging[cycle][1] for cycle in refmac_iterations]

    # convert to floats 
    refmac_iterations = [float(i) for i in refmac_iterations]
    fsc_average_curve_halfmap1_without_averaging = [float(i) for i in fsc_average_curve_halfmap1_without_averaging]
    fsc_average_curve_halfmap2_without_averaging = [float(i) for i in fsc_average_curve_halfmap2_without_averaging]
    fsc_average_curve_halfmap1_with_averaging = [float(i) for i in fsc_average_curve_halfmap1_with_averaging]
    fsc_average_curve_halfmap2_with_averaging = [float(i) for i in fsc_average_curve_halfmap2_with_averaging]

    ## Plotting
    figsize_mm = (80, 50) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
        # Plotting code here
        fig, ax = plt.subplots(1, 2, figsize=figsize_in, dpi=600)
        # set font size to font
        fontsize = 8
        plt.rcParams.update({'font.size': fontsize})
        yticks = [0.5, 0.55, 0.6]
        xticks = [0, 25, 50]
        print(len(refmac_iterations), refmac_iterations[0])
        print(len(fsc_average_curve_halfmap1_without_averaging), fsc_average_curve_halfmap1_without_averaging[0])
        print(len(fsc_average_curve_halfmap2_without_averaging), fsc_average_curve_halfmap2_without_averaging[0])
        print(len(fsc_average_curve_halfmap1_with_averaging), fsc_average_curve_halfmap1_with_averaging[0])
        print(len(fsc_average_curve_halfmap2_with_averaging), fsc_average_curve_halfmap2_with_averaging[0])
        
        print(refmac_iterations[0], refmac_iterations[25], refmac_iterations[49])
        print(fsc_average_curve_halfmap1_without_averaging[0], fsc_average_curve_halfmap1_without_averaging[25], fsc_average_curve_halfmap1_without_averaging[49])
        print(fsc_average_curve_halfmap2_without_averaging[0], fsc_average_curve_halfmap2_without_averaging[25], fsc_average_curve_halfmap2_without_averaging[49])
        print(fsc_average_curve_halfmap1_with_averaging[0], fsc_average_curve_halfmap1_with_averaging[25], fsc_average_curve_halfmap1_with_averaging[49])
        print(fsc_average_curve_halfmap2_with_averaging[0], fsc_average_curve_halfmap2_with_averaging[25], fsc_average_curve_halfmap2_with_averaging[49])

        ax[0].plot(refmac_iterations, fsc_average_curve_halfmap1_without_averaging, label="Halfmap 1")
        ax[0].plot(refmac_iterations, fsc_average_curve_halfmap2_without_averaging, label="Halfmap 2")
        ax[0].set_xlabel("Refmac cycle")
        ax[0].set_ylabel("FSC")
        ax[0].set_title("Without averaging")
        ax[0].set_ylim([0.48, 0.62])
        #ax[0].set_yticks(yticks, fontsize=fontsize)
        ax[0].set_xticks(xticks, fontsize=fontsize)
        ax[0].set_yticks(yticks, fontsize=fontsize)

        ax[1].plot(refmac_iterations, fsc_average_curve_halfmap1_with_averaging, label="Halfmap 1")
        ax[1].plot(refmac_iterations, fsc_average_curve_halfmap2_with_averaging, label="Halfmap 2")
        ax[1].set_xlabel("Refmac cycle")
        ax[1].set_ylabel("FSC")
        ax[1].set_title("With averaging")
        ax[1].set_ylim([0.48, 0.62])
        # hide y axis
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_xticks(xticks, fontsize=fontsize)
        ax[1].set_yticks(yticks, fontsize=fontsize)
        # ax[1].legend()

        fig.tight_layout()

        fig.savefig(plot_output_filename)

    print(f"Plot saved to {plot_output_filename}. Please check.")

if __name__ == "__main__":
    main()

