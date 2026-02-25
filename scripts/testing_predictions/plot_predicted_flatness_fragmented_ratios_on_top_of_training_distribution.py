## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pickle
import json
import pandas as pd
# Custom imports
from utils.chapter_3_functions import get_cumulative_probability_threshold_levels, get_2d_jointplot_with_list_of_series
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
    predicted_flatness_fragmentation_ratios_path = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "threshold_analysis", "updated_threshold_analysis_flatness_fragmentation_f1.json")
    plot_output_folder = os.path.join(data_archive_path,  "figures_output", "3_surfer", "figures", "figure_5")
    
    assert_paths_exist(flatness_fragmentation_ratios_path, predicted_flatness_fragmentation_ratios_path)
    create_folders_if_they_do_not_exist(plot_output_folder)
    
    output_filename = os.path.join(plot_output_folder, "predicted_ratios_on_training_distribution_trajectory_8958.pdf")

    # Load the training data features
    with open(flatness_fragmentation_ratios_path, 'rb') as f:
        flatness_fragmentation_ratios = pickle.load(f)
    
    # Load the predicted flatness and fragmentation ratios
    with open(predicted_flatness_fragmentation_ratios_path, 'r') as f:
        predicted_flatness_fragmentation_ratios = json.load(f)
    
    all_flatness_ratios = [float(x) for x in flatness_fragmentation_ratios["all_flatness_ratios"]]
    all_fragmentation_ratios = [float(x) for x in flatness_fragmentation_ratios["all_fragmentation_ratios"]]
    all_emdb_ids = [x for x in flatness_fragmentation_ratios.keys() if x != "all_flatness_ratios" and x != "all_fragmentation_ratios"]
    print(f"There are {len(all_emdb_ids)} entries in the training data. Expected 170.")
    # 2d kde plot for surface volume ratio (y) and flatness (x)

    list_of_flatness_ratios = []
    list_of_fragmentation_ratios = []
    list_of_f1_scores = []
    #emdb_list_for_plot = list(predicted_flatness_fragmentation_ratios.keys())
    emdb_list_for_plot = [8958]

    for chosen_emdb_id in emdb_list_for_plot:
        # Get data for plotting the series
        flatness_ratio_series = []
        fragmentation_ratio_series = []
        f1_score_series = []
        
        for threshold in predicted_flatness_fragmentation_ratios[str(chosen_emdb_id)]:
            flatness_ratio_series.append(predicted_flatness_fragmentation_ratios[str(chosen_emdb_id)][threshold]["flatness"])
            fragmentation_ratio_series.append(predicted_flatness_fragmentation_ratios[str(chosen_emdb_id)][threshold]["fragmentation"])
            f1_score_series.append(predicted_flatness_fragmentation_ratios[str(chosen_emdb_id)][threshold]["f1_score"])

        list_of_flatness_ratios.append(flatness_ratio_series)
        list_of_fragmentation_ratios.append(fragmentation_ratio_series)
        list_of_f1_scores.append(f1_score_series)

    figsize_mm = (30, 50)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        get_2d_jointplot_with_list_of_series(
            all_flatness_ratios, all_fragmentation_ratios, list_of_flatness_ratios, list_of_fragmentation_ratios, list_of_f1_scores, \
            emdb_id_list = list(predicted_flatness_fragmentation_ratios.keys()), x_label="Flatness ratio", y_label="Fragmentation ratio",
            save_path=output_filename, figsize_mm=figsize_mm, \
            probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9], 
            fontsize=10, 
            xticks=[0.2, 0.4, 0.6, 0.8], yticks=[0.2, 0.4, 0.6, 0.8], \
            cmin=0.6, cmax=1, get_trajectory=True\
            )

    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

