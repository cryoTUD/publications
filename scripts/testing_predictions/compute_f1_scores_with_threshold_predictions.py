## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.chapter_3_functions import calculate_radiomic_features
from matplotlib import rcParams
from tqdm import tqdm
# LocScale imports
sys.path.append("/home/abharadwaj1/dev/locscale")
from locscale.include.emmer.ndimage.map_utils import load_map

# Set the seed for reproducibility
np.random.seed(42)

# Global variables
emdb_ids = ["3885", "0257", "4032","4272","4288","4588","4589","4646","4746","7009","7127","7133","7882","8702","8958","8960","9934","9939"]
thresholds_for_plotting = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    predictions_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "predictions_model_20250208_103426_8")
    targets_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "micelle_remove_extra_floating_objects_test_set_low_pass")
    confidence_mask_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "test_dataset", "confidence_mask")
    output_folder = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "threshold_analysis")
    parameters_setting_file_radiomics = os.path.join(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"], "utils", "params.yml")
    assert_paths_exist(predictions_folder, targets_folder, confidence_mask_folder)
    create_folders_if_they_do_not_exist(output_folder)

    output_filename = os.path.join(output_folder, "threshold_analysis_for_predictions.pickle")
    
    # Define functions
    get_predictions_file_path = lambda x: os.path.join(predictions_folder, f"pred_model_20250208_103426_8_EMD_{int(x)}_unsharpened_fullmap.mrc.mrc")
    get_targets_file_path = lambda x: os.path.join(targets_folder, f"emd_{x}_cleaned_micelle.mrc")
    get_confidence_file_path = lambda x: os.path.join(confidence_mask_folder, f"emd_{x}_FDR_confidence_final.map")

    # for all emdb ids, compute the f1 scores for different thresholds
    f1_scores_with_thresholds_all_emdb = {}
    flatness_ratio_with_thresholds_all_emdb = {}
    fragmentation_ratio_with_thresholds_all_emdb = {}
    for emdb in tqdm(emdb_ids, desc="Processing EMDB IDs"):
        f1_scores_with_thresholds = {}
        flatness_ratio_with_thresholds = {}
        fragmentation_ratio_with_thresholds = {}

        predictions_path = get_predictions_file_path(emdb)
        targets_path = get_targets_file_path(emdb)
        confidence_path = get_confidence_file_path(emdb)

        if not os.path.exists(predictions_path) or not os.path.exists(targets_path) or not os.path.exists(confidence_path):
            continue
        predictions, apix = load_map(predictions_path)
        targets, _ = load_map(targets_path)
        confidence, _ = load_map(confidence_path)
        confidence_mask = (confidence >= 0.99).astype(bool)
        target_binary = (targets >= 0.5).astype(bool)
        target_binary_masked = target_binary[confidence_mask]
        # compute f1 score
        # precisions, recalls, thresholds = precision_recall_curve(y_true=target_binary.flatten(), probas_pred=predictions.flatten())
        # indices_in_thresholds_closest_to_thresholds_for_plotting = [np.argmin(np.abs(thresholds - threshold)) for threshold in thresholds_for_plotting]
        # precisions_for_plotting = precisions[indices_in_thresholds_closest_to_thresholds_for_plotting]
        # recalls_for_plotting = recalls[indices_in_thresholds_closest_to_thresholds_for_plotting]
        # f1_scores_for_plotting = 2 * (precisions_for_plotting * recalls_for_plotting) / (precisions_for_plotting + recalls_for_plotting)
        # # compute f1 scores
        # f1_scores_with_thresholds_all_emdb[emdb] = f1_scores_for_plotting
        for threshold in thresholds_for_plotting:
            print(f"Processing EMDB ID: {emdb} at threshold: {threshold}")
            prediction_binarised = (predictions >= threshold).astype(bool)
            prediction_binarised_masked = prediction_binarised[confidence_mask]
            f1_score_value = f1_score(y_true=target_binary_masked.flatten(), y_pred=prediction_binarised_masked.flatten())
            
            
            # compute radiomic features
            try:
                print("Computing radiomic features")
                featuredict = calculate_radiomic_features(prediction_binarised.astype(float), apix, settings_file=parameters_setting_file_radiomics)
                length = featuredict['original_shape_MajorAxisLength']
                height = featuredict['original_shape_LeastAxisLength']
                fragmentation_ratio = featuredict['original_shape_SurfaceVolumeRatio']
                flatness_ratio = 1 - height / length
            except:
                length = np.nan
                height = np.nan
                fragmentation_ratio = np.nan
                flatness_ratio = np.nan 
                print(f"Error in computing radiomic features for EMDB ID: {emdb} at threshold: {threshold}")
            # store the values
            f1_scores_with_thresholds[threshold] = f1_score_value
            flatness_ratio_with_thresholds[threshold] = flatness_ratio
            fragmentation_ratio_with_thresholds[threshold] = fragmentation_ratio

            print(f"{emdb}: {threshold} -> F1 Score: {f1_score_value}, Flatness Ratio: {flatness_ratio}, Fragmentation Ratio: {fragmentation_ratio}")
            

        f1_scores_with_thresholds_all_emdb[emdb] = f1_scores_with_thresholds
        flatness_ratio_with_thresholds_all_emdb[emdb] = flatness_ratio_with_thresholds
        fragmentation_ratio_with_thresholds_all_emdb[emdb] = fragmentation_ratio_with_thresholds


    threshold_analysis_for_predictions = {
        "f1_scores": f1_scores_with_thresholds_all_emdb,
        "flatness_ratio": flatness_ratio_with_thresholds_all_emdb,
        "fragmentation_ratio": fragmentation_ratio_with_thresholds,
        "thresholds_for_plotting": thresholds_for_plotting,
        "emdb_ids_original": emdb_ids, 
        "emdb_ids_final" : list(f1_scores_with_thresholds_all_emdb.keys()),
    } 


    # Save the values
    pd.to_pickle(threshold_analysis_for_predictions, output_filename)

if __name__ == "__main__":
    main()
