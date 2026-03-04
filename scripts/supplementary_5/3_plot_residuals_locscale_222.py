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
from sklearn.isotonic import IsotonicRegression
#from netcal.metrics import ECE, MCE
# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, plot_binned_residuals, pretty_lineplot_XY
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.emmernet.emmernet_functions import load_smoothened_mask

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

num_samples = 15
num_bins = 64
low_variance_threshold = 0.000002
#ece_metric = ECE(bins=num_bins)
#mce_metric = MCE(bins=num_bins)
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "maps")

    data_input_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "residuals_analysis_new_dataset_locscale_222")
    input_filename = os.path.join(data_input_folder_main, "residuals_analysis_new_dataset_locscale_222.pickle")
    output_folder = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "residuals_analysis_new_dataset_locscale_222")
    output_filename = os.path.join(output_folder, "regressor_isotonic_seed_42.pickle")
    #output_filename_v1 = os.path.join(output_folder, "regressor_isotonic_v1.pickle")

    output_filename_train_data = os.path.join(output_folder, "regression_train_test_data.pickle")

    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_5", "residuals_analysis_new_dataset_locscale_222")
    # other output folder
    assert_paths_exist(data_input_folder_main)
    create_folders_if_they_do_not_exist(plot_output_folder)
    uncalibrated_residual_plot_path = os.path.join(plot_output_folder, "uncalibrated_residuals_2.pdf")
    calibrated_residual_plot_path = os.path.join(plot_output_folder, "calibrated_residuals_2.pdf")
    calibration_plot_path = os.path.join(plot_output_folder, "calibration_plot_2.pdf")
    
    # Load the data
    with open(input_filename, "rb") as f:
        data = pickle.load(f)

    with open(output_filename_train_data, "rb") as f:
        train_data = pickle.load(f)

    # Load the calibrator 
    with open(output_filename, "rb") as f:
        isotonic_regressor = pickle.load(f)
    
    # with open(output_filename_v1, "rb") as f:
    #     isotonic_regressor_v1 = pickle.load(f)

    residuals_emdb = data["residuals_emdb"]
    squared_residuals_emdb = data["squared_residuals_emdb"]
    masked_variance_emdb = data["masked_variance_emdb"]

    residuals_list = list(residuals_emdb.values())
    masked_variance_list = list(masked_variance_emdb.values())
    residuals_array = np.concatenate([x for x in residuals_list])
    masked_variance_array = np.concatenate([x for x in masked_variance_list])
    
    standard_errors = np.sqrt(masked_variance_array) / np.sqrt(num_samples)
    absolute_residuals = np.abs(residuals_array)

    noise_voxels = (masked_variance_array < low_variance_threshold)
    standard_error_signal = standard_errors[~noise_voxels]
    absolute_residual_signal = absolute_residuals[~noise_voxels]

    X_test = train_data["X_test"]
    Y_test = train_data["Y_test"]

    print(f"Min and max of uncalibrated SE in test set: {X_test.min()}, {X_test.max()}")
    print(f"Min and max of absolute residuals in test set: {Y_test.min()}, {Y_test.max()}")
    # Calibrate the residuals
    calibrated_residuals = isotonic_regressor.predict(X_test)

    figsize_mm = (50, 50)
    figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
    fontsize = 8
    rcparams = configure_plot_scaling(figsize_mm, fontsize)
    with temporary_rcparams(rcparams):
        plot_binned_residuals(standard_error_signal, absolute_residual_signal, num_bins,\
        xlabel="Standard Error", ylabel="Mean Absolute Residual",\
        save_path=uncalibrated_residual_plot_path, figsize_cm=figsize_cm,\
        linewidth=0.5, marker=".", markersize=1)
    
    # Calibrate the residuals
    with temporary_rcparams(rcparams):
        plot_binned_residuals(calibrated_residuals, Y_test, num_bins,\
        xlabel="Standard Error", ylabel="Mean Absolute Residual",\
        save_path=calibrated_residual_plot_path, figsize_cm=figsize_cm,\
        linewidth=0.5, marker=".", markersize=1)
    
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        x_array = np.linspace(-0.005, 0.3, 1000)
        y_array = isotonic_regressor.predict(x_array)
        fig = pretty_lineplot_XY(x_array, y_array,\
                                xlabel="Uncalibrated SE", ylabel="Calibrated SE",
                                figsize_mm=figsize_mm, linewidth=0.5, marker=None)
        fig.savefig(calibration_plot_path, bbox_inches="tight", dpi=600)
    
    # Save plot data as json
    plot_data = {
        "uncalibrated": {
            "standard_error_signal": standard_error_signal.tolist(),
            "absolute_residual_signal": absolute_residual_signal.tolist(),
        },
        "calibrated": {
            "calibrated_residuals": calibrated_residuals.tolist(),
            "Y_test": Y_test.tolist(),
        },
        "calibration_curve": {
            "x_array": x_array.tolist(),
            "y_array": y_array.tolist(),
        }
    }
    plot_data_json_path = os.path.join(plot_output_folder, "residuals_plots_data.pickle")
    
    with open(plot_data_json_path, "wb") as f:
        pickle.dump(plot_data, f)



    print(f"Saved the plot data to {plot_data_json_path}")
    print(f"Saved the calibration plot to {calibration_plot_path}")
    print(f"Saved the uncalibrated residuals plot to {uncalibrated_residual_plot_path}")
    print(f"Saved the calibrated residuals plot to {calibrated_residual_plot_path}")

if __name__ == "__main__":
    main()

