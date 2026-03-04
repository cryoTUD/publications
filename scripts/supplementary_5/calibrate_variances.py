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
from netcal.metrics import ECE, MCE
# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, plot_binned_residuals
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.emmernet.emmernet_functions import load_smoothened_mask

# Set the seed for reproducibility
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
np.random.seed(seed)
random.seed(seed)

num_samples = 15
num_bins = 64
ece_metric = ECE(bins=num_bins)
mce_metric = MCE(bins=num_bins)
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "maps")

    data_input_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "residuals_analysis")
    input_filename = os.path.join(data_input_folder_main, "residuals_analysis_redo.pickle")
    output_folder = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "residuals_analysis")
    output_filename = os.path.join(output_folder, f"regressor_isotonic_seed_{seed}.pickle")
    output_filename_train_data = os.path.join(output_folder, "regression_train_test_data.pickle")

    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(data_input_folder_main)
    create_folders_if_they_do_not_exist(output_folder)
    
    
    # Load the data
    with open(input_filename, "rb") as f:
        data = pickle.load(f)

    residuals_emdb = data["residuals_emdb"]
    squared_residuals_emdb = data["squared_residuals_emdb"]
    masked_variance_emdb = data["masked_variance_emdb"]

    residuals_list = list(residuals_emdb.values())
    masked_variance_list = list(masked_variance_emdb.values())
    residuals_array = np.concatenate([x for x in residuals_list])
    masked_variance_array = np.concatenate([x for x in masked_variance_list])
    
    # Initialize the isotonic regression model
    isotonic_regressor = IsotonicRegression(out_of_bounds="clip")

    standard_errors = np.sqrt(masked_variance_array) / np.sqrt(num_samples)
    absolute_residuals = np.abs(residuals_array)

    X = np.array(standard_errors)
    Y = np.array(absolute_residuals)

    shuffle_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffle_indices]
    Y_shuffled = Y[shuffle_indices]

    split_index = int(len(X_shuffled) * 0.6)
    X_train = X_shuffled[:split_index]
    Y_train = Y_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    Y_test = Y_shuffled[split_index:]

    # Fit the isotonic regression model
    isotonic_regressor.fit(X_train, Y_train)
    # Predict the test set
    Y_pred = isotonic_regressor.predict(X_test)


    # save the model
    with open(output_filename, "wb") as f:
        pickle.dump(isotonic_regressor, f)

    # save the test and train data
    train_data = {
        "X_test": X_test,
        "Y_test": Y_test,
        "X_train": X_train,
        "Y_train": Y_train
    }

    with open(output_filename_train_data, "wb") as f:
        pickle.dump(train_data, f)

    print(f"Regression model saved to {output_filename}. Please check.")
    print(f"Train and test data saved to {output_filename_train_data}. Please check.")

if __name__ == "__main__":
    main()

