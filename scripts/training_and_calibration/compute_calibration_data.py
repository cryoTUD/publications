## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pandas as pd
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams


# LocScale imports
from locscale.include.emmer.ndimage.map_utils import load_map

# Set the seed for reproducibility
np.random.seed(42)

# Global variables
emdb_ids = ["0257", "4032","4272","4288","4588","4589","4646","4746","7009","7133","7882","8702","8958","8960","9934","9939"]
n_bins = 50

def compute_ece(probabilities, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) by binning the predictions.
    
    Parameters:
        probabilities (np.array): Array of predicted probabilities.
        labels (np.array): True binary labels (0 or 1).
        n_bins (int): Number of bins.
    
    Returns:
        float: Expected Calibration Error.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    total = len(probabilities)
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i+1]
        mask = (probabilities >= lower) & (probabilities < upper)
        if np.sum(mask) > 0:
            avg_conf = np.mean(probabilities[mask])
            avg_acc = np.mean(labels[mask])
            ece += np.abs(avg_conf - avg_acc) * np.sum(mask) / total
    return ece

## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    predictions_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "predictions_model_20250208_103426_8")
    targets_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "micelle_remove_extra_floating_objects_test_set_low_pass")
    confidence_mask_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "confidence_mask")
    output_folder = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "calibration_analysis")
    plot_output_folder = os.path.join(data_archive_path, "figures_output", "3_surfer", "figures", "figure_4")
    
    assert_paths_exist(predictions_folder, targets_folder, confidence_mask_folder)
    create_folders_if_they_do_not_exist(output_folder, plot_output_folder)

    output_filename = os.path.join(output_folder, "calibration_curve_data.pickle")
    plot_filename = os.path.join(plot_output_folder, "calibration_curve.pdf")
    # Define functions
    get_predictions_file_path = lambda x: os.path.join(predictions_folder, f"pred_model_20250208_103426_8_EMD_{int(x)}_unsharpened_fullmap.mrc.mrc")
    get_targets_file_path = lambda x: os.path.join(targets_folder, f"emd_{x}_cleaned_micelle.mrc")
    get_confidence_file_path = lambda x: os.path.join(confidence_mask_folder, f"emd_{x}_FDR_confidence_final.map")

    filenames_dictionary = { 
        emdb_id: {
            "prediction_map": get_predictions_file_path(emdb_id),
            "target_map": get_targets_file_path(emdb_id),
            "confidence_mask": get_confidence_file_path(emdb_id)
        } for emdb_id in emdb_ids
    }

    # Split
    emdb_id_list = list(filenames_dictionary.keys())
    print("Calibration IDs:", emdb_id_list)

    # TESTING: Evaluate the calibration on the test set.
    test_scores_list = []
    test_labels_list = []

    for emdb_id in emdb_id_list:
        file_dict = filenames_dictionary[emdb_id]

        conf_mask, _ = load_map(file_dict["confidence_mask"])
        target_map, _ = load_map(file_dict["target_map"])
        pred_map, _ = load_map(file_dict["prediction_map"])

        target_bin = (target_map > 0.5).astype(np.int32)
        # Use a looser mask for testing.
        mask = conf_mask > 0.5
        scores = pred_map[mask].flatten()
        labels = target_bin[mask].flatten()
        # Clip scores to the range [0, 1]
        scores = np.clip(scores, 0.0, 1.0)
        test_scores_list.append(scores)
        test_labels_list.append(labels)

    test_all_scores = np.concatenate(test_scores_list)
    test_all_labels = np.concatenate(test_labels_list)

    brier_score = brier_score_loss(test_all_labels, test_all_scores)
    print(f"Brier score on the test set: {brier_score:.4f}")
    ece = compute_ece(test_all_scores, test_all_labels, n_bins=n_bins)
    print(f"Expected Calibration Error on the test set: {ece:.4f}")

    # Compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(test_all_labels, test_all_scores, n_bins=n_bins)

    # Save the values
    calibration_data = {
        "test_all_labels": test_all_labels,
        "test_all_scores": test_all_scores,
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "n_bins": n_bins,
        "brier_score": brier_score,
        "ece": ece,
    }

    pd.to_pickle(calibration_data, output_filename)

    # plot 2 subplots in a single figure showing the calibration curve and the difference between predicted confidence and ideal confidence
    
    # Plot the calibration curve
    figsize_mm = (80, 50)
    rc_params = configure_plot_scaling(figsize_mm)
    #rc_params["axes.labelsize"] = 8
    with temporary_rcparams(rc_params):
        # Make the plot editable in Illustrator
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
        sns.set_theme(context="paper")
        calibration_error = mean_predicted_value - fraction_of_positives
        ax.plot(mean_predicted_value, fraction_of_positives, marker=".", label="Calibration curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal calibration")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Fraction of positives")
        # add secondary y-axis for calibration error
        ax2 = ax.twinx()
        ax2.set_ylabel("Calibration error", color="red")
        # axis labels in red
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.plot(mean_predicted_value, calibration_error, color="red", label="Calibration error")
        ax2.axhline(0, color="red", linestyle="--", linewidth=0.5)
        
        #ax.legend()

        fig.tight_layout()
        fig.savefig(plot_filename)
        plt.close(fig)

    print(f"Calibration curve data saved to {output_filename}")
    print(f"Calibration curve plot saved to {plot_filename}")
if __name__ == "__main__":
    main()
