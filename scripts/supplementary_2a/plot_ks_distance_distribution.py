## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd

from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "ks_distances.csv")  # one mode analysis permutation_test_results_adp_correlations.pickle
    output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a")
    output_path = os.path.join(output_folder, "ks_distance_distribution_2_mode.pdf")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    # with open(input_path, "rb") as f:
    #     data = pickle.load(f)
    data = pd.read_csv(input_path)

    ks_distance_pseudo_values = data["ks_distance_pseudo_values"]
    ks_distance_atomic_values = data["ks_distance_atomic_values"]
    ks_pvalues_pseudo_values = data["ks_pvalues_pseudo_values"]
    ks_pvalues_atomic_values = data["ks_pvalues_atomic_values"]
    emdb_pdbs = data["EMDB_PDB_ids_present"]
    # print pvalues range 
    print(f"Pseudo-atomic model KS p-values range: {ks_pvalues_pseudo_values.min()} - {ks_pvalues_pseudo_values.max()}")
    print(f"Atomic model KS p-values range: {ks_pvalues_atomic_values.min()} - {ks_pvalues_atomic_values.max()}")

    # Create a dataframe to hold the values
    df_pseudo = pd.DataFrame({
        "KS Distance": ks_distance_pseudo_values,
        "P-value": ks_pvalues_pseudo_values,
        "Type": "Pseudo-atomic model"
    })
    df_atomic = pd.DataFrame({
        "KS Distance": ks_distance_atomic_values,
        "P-value": ks_pvalues_atomic_values,
        "Type": "Atomic model"
    })

    # Combine the dataframes
    df_combined = pd.concat([df_pseudo, df_atomic]) 
    
    # Colors 
    norm_pseudo = plt.Normalize(vmin=min(ks_distance_pseudo_values), vmax=max(ks_distance_pseudo_values))
    norm_atomic = plt.Normalize(vmin=min(ks_distance_atomic_values), vmax=max(ks_distance_atomic_values))

    # Color map
    cmap = sns.color_palette("rainbow", as_cmap=True)
    colors_pseudo = [cmap(norm_pseudo(x)) for x in ks_distance_pseudo_values]
    colors_atomic = [cmap(norm_atomic(x)) for x in ks_distance_atomic_values]

    # Plot configuration
    figsize_mm = (75, 100)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(1, 1, figsize=figsize_in, dpi=600)
        sns.boxplot(x="Type", y="KS Distance", data=df_combined, ax=ax, color="white")
        sns.swarmplot(x="Type", y="KS Distance", data=df_combined, ax=ax, hue="P-value")

        # Set the ticks 
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pseudo-atomic\nmodel", "Atomic\nmodel"])
        ax.set_ylabel("KS Distance")
        #ax.set_yticks([0, 0.1, 0.2])

        fig.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")

    print(f"Saved scatter plot to {output_path}")

if __name__ == "__main__":
    main()