## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams
from tqdm import tqdm
# LocScale imports
from locscale.include.emmer.ndimage.map_utils import load_map

# Set the seed for reproducibility
np.random.seed(42)

# Global variables
emdb_ids = ["0257", "4032","4272","4288","4588","4589","4646","4746","7009","7127","7133","7882","8702","8958","8960","9934","9939"]
## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    predictions_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "predictions_model_20250208_103426_8")
    targets_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "micelle_remove_extra_floating_objects_test_set_low_pass")
    confidence_mask_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "confidence_mask")
    output_folder = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "threshold_analysis")
    
    assert_paths_exist(predictions_folder, targets_folder, confidence_mask_folder)
    create_folders_if_they_do_not_exist(output_folder)

    output_filename = os.path.join(output_folder, "sigmoid_output_distribution.pickle")
    
    # Define functions
    get_predictions_file_path = lambda x: os.path.join(predictions_folder, f"pred_model_20250208_103426_8_EMD_{int(x)}_unsharpened_fullmap.mrc.mrc")
    get_targets_file_path = lambda x: os.path.join(targets_folder, f"emd_{x}_cleaned_micelle.mrc")
    get_confidence_file_path = lambda x: os.path.join(confidence_mask_folder, f"emd_{x}_FDR_confidence_final.map")

    # for all emdb ids, get the values inside the confidence mask but outside the targets, inside targets and plot the distribution of values
    values_inside_confidence_mask_outside_targets_all = {}
    values_inside_targets_all = {}
    values_inside_confidence_mask_all = {}

    for emdb in tqdm(emdb_ids, desc="Processing EMDB IDs"):
        predictions_path = get_predictions_file_path(emdb)
        targets_path = get_targets_file_path(emdb)
        confidence_path = get_confidence_file_path(emdb)

        if not os.path.exists(predictions_path) or not os.path.exists(targets_path) or not os.path.exists(confidence_path):
            continue
        predictions, apix = load_map(predictions_path)
        targets, _ = load_map(targets_path)
        confidence, _ = load_map(confidence_path)

        inside_confidence_mask = (confidence >= 0.99).astype(bool)
        inside_targets = (targets >= 0.5).astype(bool)
        inside_confidence_mask_outside_targets = inside_confidence_mask & ~inside_targets

        values_inside_confidence_mask = predictions[inside_confidence_mask]
        values_inside_targets = predictions[inside_targets]
        values_inside_confidence_mask_outside_targets = predictions[inside_confidence_mask_outside_targets]

        values_inside_confidence_mask_outside_targets_all[emdb] = values_inside_confidence_mask_outside_targets
        values_inside_targets_all[emdb] = values_inside_targets
        values_inside_confidence_mask_all[emdb] = values_inside_confidence_mask

    # Save the values
    output_distributions_all_voxels = {
        "values_inside_confidence_mask_outside_targets": values_inside_confidence_mask_outside_targets_all,
        "values_inside_targets": values_inside_targets_all,
        "values_inside_confidence_mask": values_inside_confidence_mask_all
    }

    pd.to_pickle(output_distributions_all_voxels, output_filename)

if __name__ == "__main__":
    main()
