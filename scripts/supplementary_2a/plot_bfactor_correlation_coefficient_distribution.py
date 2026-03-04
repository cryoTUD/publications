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

from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "atomic_v_pseudoatomic_adp_correlation_filtered.pickle")
    output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a")
    output_path = os.path.join(output_folder, "bfactor_correlation_pseudo_vs_atomic_scatter.pdf")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    emdb_pdbs = data["EMDB_PDB_ids_present"]
    bfactor_list_atomic = data["bfactor_list_atomic"]
    bfactor_list_pseudo = data["bfactor_list_pseudo"]
    correlation_dict = data["bfactor_correlation_emdb"]

    correlation_values = [x[0] for x in correlation_dict.values()]
    correlation_values = np.array(correlation_values)
    # Plot configuration
    figsize_mm = (70, 60)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(1, 1, figsize=figsize_in, dpi=600)
        sns.histplot(correlation_values, stat="density", ax=ax, bins=30)
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Frequency")
        ax.set_title("B-factor correlation")
        text = f"N = {len(correlation_values)}"
        ax.text(0.2, 0.9, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)

        ax.set_xlim(0,1)
        fig.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")

    print(f"Saved scatter plot to {output_path}")

    # print EMDB PDB with lowest five correlation values
    sorted_correlation = sorted(correlation_dict.items(), key=lambda x: x[1][0])
    print("Lowest five correlation values:")
    for emdb_pdb, corr in sorted_correlation[:5]:
        print(f"{emdb_pdb}: {corr[0]}")

if __name__ == "__main__":
    main()
