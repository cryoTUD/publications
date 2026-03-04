## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 

from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams

np.random.seed(42)
random.seed(42)

def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "figure_7", "model_angelo_residue_counts.pickle")
    output_folder = os.path.join(data_archive_path, "outputs", "figure_7")
    output_path = os.path.join(output_folder, "difference_in_number_of_residues_before_pruning.pdf")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    with open(input_path, "rb") as f:
        model_angelo_results = pickle.load(f)

    raw_counts_hybrid = model_angelo_results["num_residues_using_hybrid_raw"]
    raw_counts_unsharpened = model_angelo_results["num_residues_using_unsharpened_raw"]

    differences = []

    for key in raw_counts_hybrid:
        if key in raw_counts_unsharpened:
            diff = raw_counts_hybrid[key] - raw_counts_unsharpened[key]
            differences.append(diff)

    figsize_mm = (70, 50)
    rcparams = configure_plot_scaling(figsize_mm)

    with temporary_rcparams(rcparams):
        fig, ax = plt.subplots()
        sns.histplot(differences, bins=30, kde=False, ax=ax, color="darkblue")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Residue Count Difference Before Pruning\n(Hybrid - Unsharpened)")
        ax.set_xlabel("Difference in number of residues")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")

    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
