## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
sys.path.append("/home/abharadwaj1/dev/locscale")
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
from locscale.include.emmer.pdb.pdb_utils import get_bfactors
from s2a_utils import plot_invgamma_fit, get_atomic_bfactor_correlation

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "atomic_v_pseudoatomic_adp_correlation_filtered.pickle")
    output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a")
    output_path_3061 = os.path.join(output_folder, "adp_distribution_3061.pdf")
    output_path_8702 = os.path.join(output_folder, "adp_distribution_8702.pdf")
    
    pseudomodel_pdb_path = "/home/abharadwaj1/papers/elife_paper/figure_information/data/maps/emd_3061/emd_3061_pseudomodel_within_atomic_mask_tight.pdb"
    atomic_model_path = "/home/abharadwaj1/papers/elife_paper/figure_information/data/maps/emd_3061/model_based/processing_files/5a63_shifted_servalcat_refined.pdb"
    
    
    assert_paths_exist(input_path, pseudomodel_pdb_path, atomic_model_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    emdb_pdbs = data["EMDB_PDB_ids_present"]
    bfactor_list_atomic = data["bfactor_list_atomic"]
    bfactor_list_pseudo = data["bfactor_list_pseudo"]
    spearman_dict = data["bfactor_correlation_emdb_spearman"]

    bfactor_comparison = get_atomic_bfactor_correlation(pseudomodel_pdb_path, atomic_model_path)
    pseudo_bfactors = np.array([x[1] for x in bfactor_comparison.values()]) 
    atomic_bfactors = np.array([x[0] for x in bfactor_comparison.values()])

    # add this to the data dict
    bfactor_list_atomic["3061"] = atomic_bfactors
    bfactor_list_pseudo["3061"] = pseudo_bfactors

    figsize_mm = (60, 30)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig = plot_invgamma_fit("3061", bfactor_list_pseudo, bfactor_list_atomic)
        fig.tight_layout()
        fig.savefig(output_path_3061, bbox_inches="tight")

    print(f"Saved scatter plot to {output_path_3061}")

    # Plot for 8702
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig = plot_invgamma_fit("8702_5vkq", bfactor_list_pseudo, bfactor_list_atomic)
        fig.tight_layout()
        fig.savefig(output_path_8702, bbox_inches="tight")

if __name__ == "__main__":
    main()