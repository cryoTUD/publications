## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling, plot_kde_ridge, plot_ridgeplot
from matplotlib import rcParams


# Set the seed for reproducibility
np.random.seed(42)

# Global variables
sample_size = 5000
## SETUP


def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    sigmoid_output_distribution_path = os.path.join(data_archive_path, \
                "processed_data_output", "3_surfer", "threshold_analysis", "sigmoid_output_distribution.pickle")
    f1_scores_prediction_path = os.path.join(data_archive_path, \
                "inputs", "3_surfer", "network_training", "test_dataset", "f1_scores_test_dataset.txt")
    output_folder = os.path.join(data_archive_path, "figures_output", "3_surfer", "figures", "figure_4")
    
    assert_paths_exist(sigmoid_output_distribution_path, f1_scores_prediction_path)
    create_folders_if_they_do_not_exist(output_folder)

    output_filename_sigmoid_distribution = os.path.join(output_folder, "sigmoid_output_distribution_inside_targets.pdf")
    output_filename_threshold_analysis = os.path.join(output_folder, "threshold_analysis.pdf")
    output_filename_optimal_threshold = os.path.join(output_folder, "optimal_threshold_distribution.pdf")

    # Load the inputs 
    df = pd.read_csv(f1_scores_prediction_path, sep=",")
    f1_scores_emdb_id = {int(x) : float(df[df['emdb_id']==x][' f1']) for x in df['emdb_id']}
    sigmoid_output_distribution = pd.read_pickle(sigmoid_output_distribution_path)
    values_inside_targets_all = sigmoid_output_distribution["values_inside_targets"]
    values_inside_confidence_mask_outside_targets_all = sigmoid_output_distribution["values_inside_confidence_mask_outside_targets"]
    values_inside_confidence_mask_all = sigmoid_output_distribution["values_inside_confidence_mask"]
    print("Values inside targets:", len(values_inside_targets_all))
    print("Values inside confidence mask outside targets:", len(values_inside_confidence_mask_outside_targets_all))
    print("Values inside confidence mask:", len(values_inside_confidence_mask_all))
    num_voxels_inside_target = 0 
    num_voxels_inside_confidence_mask_outside_target = 0
    for emdb_id in values_inside_targets_all.keys():
        num_voxels_inside_target_in_this_emdb = len(values_inside_targets_all[emdb_id])
        num_voxels_inside_confidence_mask_outside_target_in_this_emdb = len(values_inside_confidence_mask_outside_targets_all[emdb_id])
        total_voxels_inside_confidence_mask_in_this_emdb = len(values_inside_confidence_mask_all[emdb_id])
        print(f"{emdb_id} : {num_voxels_inside_target_in_this_emdb}")

              
        num_voxels_inside_target += len(values_inside_targets_all[emdb_id])
        num_voxels_inside_confidence_mask_outside_target += len(values_inside_confidence_mask_outside_targets_all[emdb_id])
    print(f"Total number of voxels inside targets: {num_voxels_inside_target}")
    print(f"Total number of voxels inside confidence mask outside targets: {num_voxels_inside_confidence_mask_outside_target}")
    total_num_voxels = num_voxels_inside_target + num_voxels_inside_confidence_mask_outside_target
    print(f"Total number of voxels: {total_num_voxels}")
    sys.exit(0)
    # Get the set of all emdb ids present in both the data
    emdb_ids_in_f1_scores = [int(x) for x in f1_scores_emdb_id.keys()]
    emdb_ids_in_sigmoid_output = [int(x) for x in values_inside_targets_all.keys()]

    #common_emdb_ids = [x for x in emdb_ids_in_sigmoid_output if int(x) in emdb_ids_in_f1_scores]
    common_emdb_ids = list(set(emdb_ids_in_f1_scores) & set(emdb_ids_in_sigmoid_output))

    print(f"Number of EMDB IDs: {len(common_emdb_ids)}")
    all_emdb_kdes = {}
    sigmoid_output_values = np.linspace(0, 1, 1000)
    means_emdb = {}
    for emdb in tqdm(values_inside_confidence_mask_all, desc="Extracting values"):
        values_inside_confidence_mask_in_this_emdb = values_inside_confidence_mask_all[emdb]
        values_inside_targets_in_this_emdb = values_inside_targets_all[emdb]
        values_inside_confidence_mask_outside_targets_in_this_emdb = values_inside_confidence_mask_outside_targets_all[emdb]
        random_sample_inside_targets = np.random.choice(values_inside_targets_in_this_emdb, sample_size//2)
        random_sample_outside_targets = np.random.choice(values_inside_confidence_mask_outside_targets_in_this_emdb, sample_size//2)
        random_sample_of_values = np.concatenate((random_sample_inside_targets, random_sample_outside_targets))

        #random_sample_of_values = np.random.choice(values_inside_targets_in_this_emdb, sample_size)
        means_emdb[emdb] = np.mean(random_sample_of_values)
        # Fit the kernel density
        kde = gaussian_kde(random_sample_of_values)
        # Compute the density on a grid
        kde.set_bandwidth(bw_method="silverman")
        kde_values = kde(sigmoid_output_values)
        all_emdb_kdes[emdb] = kde_values
    
    # Plot the training loss curve
    figsize_mm = (50, 80)
    rc_params = configure_plot_scaling(figsize_mm)
    rc_params["axes.labelsize"] = 8
    with temporary_rcparams(rc_params):
        # Make the plot editable in Illustrator
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
        sns.set_theme(context="paper")

        # Plot the KDEs using the ridge plot
        f1_scores_dictionary = {int(x): f1_scores_emdb_id[x] for x in f1_scores_emdb_id.keys()}
        kde_values_dictionary = {int(x): all_emdb_kdes[x] for x in all_emdb_kdes.keys()}

        ax = plot_kde_ridge(f1_scores_dictionary, kde_values_dictionary, means=means_emdb, ax=ax, \
                        clabel="Segmentation accuracy at 0.5 (F1)", xlabel="Voxel confidence", ylabel="Normalized KDE")
        
        #ax.set_xlabel("Sigmoid output")
        #ax.set_ylabel("Normalised distribution of predicted voxels")
        # Hide y axis
        #ax.get_yaxis().set_visible(False)

        # draw a dashed line at 0.5
        ax.axvline(x=0.5, linestyle="--", color="black", alpha=0.5)

        # Save the plot
        fig.tight_layout()
        fig.savefig(output_filename_sigmoid_distribution)
        plt.close(fig)

    ## Compute and plot the effect of threshold on the F1 score
        
    threshold_values = np.linspace(0, 1, 100)
    #threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    precision_values_dictionary = {}
    recall_values_dictionary = {}
    f1_values_for_threshold_analysis_dictionary = {}
    emdb_ids_for_threshold_analysis = list(values_inside_confidence_mask_all.keys())
    #emdb_ids_for_threshold_analysis = ["8702"]
    for emdb in tqdm(emdb_ids_for_threshold_analysis, desc="Computing threshold analysis"):
        values_inside_targets_in_this_emdb = values_inside_targets_all[emdb]
        values_inside_confidence_mask_outside_targets_in_this_emdb = values_inside_confidence_mask_outside_targets_all[emdb]
        
        precision_values = []
        recall_values = []
        f1_values_for_threshold_analysis = []
        
        for threshold in threshold_values:
            tp = np.sum(values_inside_targets_in_this_emdb >= threshold)
            fp = np.sum(values_inside_confidence_mask_outside_targets_in_this_emdb >= threshold)
            fn = np.sum(values_inside_targets_in_this_emdb < threshold)
            
            precision = tp / (tp + fp)  if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values_for_threshold_analysis.append(f1)
            
        
        precision_values_dictionary[emdb] = precision_values
        recall_values_dictionary[emdb] = recall_values
        f1_values_for_threshold_analysis_dictionary[emdb] = f1_values_for_threshold_analysis
    
    # Compute the optimal threshold and the corresponding F1 score for each emdb id
    optimal_thresholds = {}
    optimal_f1_scores = {}
    for emdb_id in tqdm(f1_values_for_threshold_analysis_dictionary, desc="Computing optimal threshold"):
        f1_values = f1_values_for_threshold_analysis_dictionary[emdb_id]
        optimal_threshold_index = np.argmax(f1_values)
        optimal_threshold = threshold_values[optimal_threshold_index]
        optimal_f1_score = f1_values[optimal_threshold_index]
        
        optimal_thresholds[emdb_id] = optimal_threshold
        optimal_f1_scores[emdb_id] = optimal_f1_score
    # Plot the F1 score as a function of threshold for all EMDB IDs 
    print(f"Optimal thresholds: {optimal_thresholds}")
    # Sort the EMDB IDs by Optimal F1 score in increasing order
    sorted_emdb_ids = sorted(optimal_thresholds, key=optimal_thresholds.get)

    figsize_mm = (60, 40)
    rc_params = configure_plot_scaling(figsize_mm)

    with temporary_rcparams(rc_params):
        # Make the plot editable in Illustrator
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize, dpi=600)
        sns.set_theme(context="paper")
        sns.set_style("white")
        y_offset = 0
        spacing = 0
        for i, emdb_id in enumerate(sorted_emdb_ids):
            f1_values = np.array(f1_values_for_threshold_analysis_dictionary[emdb_id])
            f1_values = f1_values + i * spacing
            if emdb_id == "8702":
                ax.plot(threshold_values, f1_values, color="red", alpha=1)
            elif emdb_id == "8958":
                ax.plot(threshold_values, f1_values, color="blue", alpha=1)
            else:
                ax.plot(threshold_values, f1_values, color="black", alpha=0.3)
            
        
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 score")
        # Hide y axis ticks and tick labels
        #ax.get_yaxis().set_visible(False)
        ax.get_yaxis().set_ticks([0, 0.5, 1])
        #ax.get_yaxis().set_ticklabels([])
        
        # Add a dot at the optimal threshold for each EMDB ID showing the optimal F1 score
        f1_scores_array_all_emdb = np.array([optimal_f1_scores[emdb_id] for emdb_id in sorted_emdb_ids])
        norm_f1_scores = (f1_scores_array_all_emdb - np.min(f1_scores_array_all_emdb)) / (np.max(f1_scores_array_all_emdb) - np.min(f1_scores_array_all_emdb))
        colors = cm.viridis(norm_f1_scores)
        
        for i, emdb_id in enumerate(sorted_emdb_ids):
            optimal_threshold = optimal_thresholds[emdb_id]
            optimal_f1_score = optimal_f1_scores[emdb_id]  + i * spacing
            if emdb_id == "8702":
                ax.scatter(optimal_threshold, optimal_f1_score, color="red", s=10)
            elif emdb_id == "8958":
                ax.scatter(optimal_threshold, optimal_f1_score, color="blue", s=10)
            else:
                ax.scatter(optimal_threshold, optimal_f1_score, color="black", s=10, alpha=0.3)
    
        # Save the plot
        fig.tight_layout()
        fig.savefig(output_filename_threshold_analysis)
        plt.close(fig)

    
    # Plot the distribution of optimal thresholds
    figsize_mm = (60, 40)
    rc_params = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rc_params):
        # Make the plot editable in Illustrator
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize, dpi=600)
        sns.set_theme(context="paper")
        sns.set_style("white")
        optimal_thresholds_array = np.array([optimal_thresholds[emdb_id] for emdb_id in optimal_thresholds.keys()])
        sns.histplot(optimal_thresholds_array, kde=True, ax=ax, bins=20)
        ax.set_xlabel("Optimal threshold")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(output_filename_optimal_threshold)
        plt.close(fig)

    print(f"Sigmoid output distribution plot saved to {output_filename_sigmoid_distribution}")
    print(f"Threshold analysis plot saved to {output_filename_threshold_analysis}")
    print(f"Optimal threshold distribution plot saved to {output_filename_optimal_threshold}")

if __name__ == "__main__":
    main()
