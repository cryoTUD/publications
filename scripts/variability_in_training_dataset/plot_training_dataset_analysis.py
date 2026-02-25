## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pickle
import pandas as pd
# Custom imports
from utils.chapter_3_functions import get_cumulative_probability_threshold_levels, get_2d_jointplot_with_text
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams
from tqdm import tqdm

# Set the seed for reproducibility
np.random.seed(42)

## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    flatness_fragmentation_ratios_path = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "training_data_analysis", "flatness_fragmentation_ratios.pickle")
    plot_output_folder = os.path.join(data_archive_path,  "figures_output", "3_surfer", "figures", "figure_3")
    assert_paths_exist(flatness_fragmentation_ratios_path)
    create_folders_if_they_do_not_exist(plot_output_folder)
    
    output_filename = os.path.join(plot_output_folder, "flatness_fragmentation_ratio_distribution.pdf")

    # Load the training data features
    with open(flatness_fragmentation_ratios_path, 'rb') as f:
        flatness_fragmentation_ratios = pickle.load(f)
    
    all_flatness_ratios = [float(x) for x in flatness_fragmentation_ratios["all_flatness_ratios"]]
    all_fragmentation_ratios = [float(x) for x in flatness_fragmentation_ratios["all_fragmentation_ratios"]]
    all_emdb_ids = [x for x in flatness_fragmentation_ratios.keys() if x != "all_flatness_ratios" and x != "all_fragmentation_ratios"]
    print(f"There are {len(all_emdb_ids)} entries in the training data. Expected 170.")
    # 2d kde plot for surface volume ratio (y) and flatness (x)
    figsize_mm = (30, 30)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        get_2d_jointplot_with_text(
            all_flatness_ratios, all_fragmentation_ratios, emdb_id_list = all_emdb_ids, \
            x_label="Flatness ratio", y_label="Fragmentation ratio", save_path=output_filename, figsize_mm=figsize_mm, \
            probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9], 
            fontsize=10, 
            xticks=[0.4, 0.6, 0.8], yticks=[0.2, 0.4, 0.6, 0.8], \
            mark_emdb_ids={
                "red" : ["0499", "0928", "13201", "25691"], 
                "blue" : ["4270", "9941", "12095", "30713"], 
                "green" : ["9696", "13880"]
                },
            )

    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

