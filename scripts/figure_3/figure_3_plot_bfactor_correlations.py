## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
# Custom imports
from scripts.figure_3.figure_3_functions import plot_correlations
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist, any_files_are_missing

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)
# Global variables
figure_number = 3
input_map_format = "EMD_{}_unsharpened_fullmap.mrc"
target_map_format = "emd_{}_hybrid_model_map_refined_version_C.mrc"
mask_format = "emd_{}_FDR_confidence_final.map"
emdb_ids = ["0282", "0311", "10365", "20220", "20226", "3545", "4571", "4997", "7127", "8702", "9610"]
n_jobs = 10
## SETUP
def main():
    # Setup environment and define paths
    data_archive_path = setup_environment()

    input_maps_folder = os.path.join(data_archive_path, "raw","maps", f"figure_{figure_number}")
    emmernet_output_dir = os.path.join(input_maps_folder, "emmernet_prediction_hybrid_model_map_60k_test_dataset")
    confidence_mask_dir = os.path.join(data_archive_path, "raw","maps", "confidence_masks")
    atomic_mask_dir = os.path.join(data_archive_path, "raw","maps", "atomic_model_mask")
    target_model_map_dir = os.path.join(data_archive_path, "raw","maps", "hybrid_model_maps_version_C")
    
    output_structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data", f"figure_{figure_number}")

    assert_paths_exist(
        input_maps_folder, emmernet_output_dir, confidence_mask_dir, atomic_mask_dir, target_model_map_dir
    )

    create_folders_if_they_do_not_exist(
        output_structured_data_folder
    )

    # Inputs
    input_file = os.path.join(output_structured_data_folder, f"figure_{figure_number}_bfactor_correlations.pickle")
    # Outputs
    plot_output_file_fdr = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", f"figure_{figure_number}_bfactor_correlations.pdf")
    plot_output_file_fdr_png = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", f"figure_{figure_number}_bfactor_correlations.png")
    plot_output_file_atomic = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", f"figure_{figure_number}_bfactor_correlations_atomic.pdf")
    plot_output_file_atomic_png = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", f"figure_{figure_number}_bfactor_correlations_atomic.png")
    
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    # Initialize dictionary for bfactor correlations
    bfactors_in_fdr_mask_all_emdb = data["bfactors_in_fdr_mask_all_emdb"]
    bfactors_in_atomic_mask_all_emdb = data["bfactors_in_atomic_mask_all_emdb"]

    bfactors_emmernet_fdr_mask = []
    bfactors_target_model_fdr_mask = []
    qfit_emmernet_fdr_mask = []
    qfit_target_model_fdr_mask = []
    bfactors_emmernet_atomic_mask = []
    bfactors_target_model_atomic_mask = []
    qfit_emmernet_atomic_mask = []
    qfit_target_model_atomic_mask = []


    for emdb_pdb in bfactors_in_fdr_mask_all_emdb:
        bfactors_emmernet_fdr_mask += [-1* bfactor[0] for bfactor in bfactors_in_fdr_mask_all_emdb[emdb_pdb]]
        bfactors_target_model_fdr_mask += [-1 * bfactor[1] for bfactor in bfactors_in_fdr_mask_all_emdb[emdb_pdb]]
        qfit_emmernet_fdr_mask += [bfactor[2] for bfactor in bfactors_in_fdr_mask_all_emdb[emdb_pdb]]
        qfit_target_model_fdr_mask += [bfactor[3] for bfactor in bfactors_in_fdr_mask_all_emdb[emdb_pdb]]

        bfactors_emmernet_atomic_mask += [-1 * bfactor[0] for bfactor in bfactors_in_atomic_mask_all_emdb[emdb_pdb]]
        bfactors_target_model_atomic_mask += [-1 * bfactor[1] for bfactor in bfactors_in_atomic_mask_all_emdb[emdb_pdb]]
        qfit_emmernet_atomic_mask += [bfactor[2] for bfactor in bfactors_in_atomic_mask_all_emdb[emdb_pdb]]
        qfit_target_model_atomic_mask += [bfactor[3] for bfactor in bfactors_in_atomic_mask_all_emdb[emdb_pdb]]

    # print min, mean, median and max qfit 
    print("Median qfit values:")
    print(f"EMMERNet FDR mask: {np.median(qfit_emmernet_fdr_mask)}")
    print(f"EMMERNet Atomic mask: {np.median(qfit_emmernet_atomic_mask)}")
    kwargs = {
        "figsize_cm":(6,4),
        "font":"Helvetica",
        "fontsize":18,
        "find_correlation":True,
        "alpha":0.3,
        "x_label":"Wilson B-factor (EMmerNet)",
        "y_label":"Wilson B-factor (Target Model)",
        "title_text":"B-factor correlation",
        "scatter":True,
        "xticks":[0, 100, 200],
        "yticks":[0, 100, 200],
        #"xlim":[35, 125],
        #"ylim":[0, 125],
    }

    plot_correlations(\
        bfactors_emmernet_fdr_mask, bfactors_target_model_fdr_mask, filepath=plot_output_file_fdr, hue_array=qfit_emmernet_fdr_mask, \
        **kwargs
    )
    
    plot_correlations(\
        bfactors_emmernet_fdr_mask, bfactors_target_model_fdr_mask, filepath=plot_output_file_fdr_png, hue_array=qfit_emmernet_fdr_mask, \
        **kwargs
    )
    
    plot_correlations(\
        bfactors_emmernet_atomic_mask, bfactors_target_model_atomic_mask, filepath=plot_output_file_atomic, hue_array=qfit_emmernet_atomic_mask, \
        **kwargs
    )
    
    plot_correlations(\
        bfactors_emmernet_atomic_mask, bfactors_target_model_atomic_mask, filepath=plot_output_file_atomic_png, hue_array=qfit_emmernet_atomic_mask, \
        **kwargs
    )
    print(f"Plot save paths: {plot_output_file_fdr}")

    # Save the distribution of qfit values as a seaborn histogram
    fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
    sns.violinplot(data=[qfit_emmernet_fdr_mask, qfit_target_model_fdr_mask], ax=ax, inner="box", cut=0)
    #sns.swarmplot(data=[qfit_emmernet_fdr_mask, qfit_target_model_fdr_mask], ax=ax, color="k", alpha=0.5)
    ax.set_xticklabels(["EMMERNet", "Target Model"])
    save_path = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", f"figure_{figure_number}_qfit_distribution_fdr_mask.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)

    
if __name__ == "__main__":
    main()
