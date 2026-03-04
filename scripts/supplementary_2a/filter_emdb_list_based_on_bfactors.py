## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
from s2a_utils import check_if_bfactors_proper

def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "atomic_v_pseudoatomic_adp_correlation.pickle")
    output_folder = os.path.join(data_archive_path, "structured_data", "supplementary_2a")
    output_path = os.path.join(output_folder, "atomic_v_pseudoatomic_adp_correlation_filtered.pickle")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    filtered_data = {}
    bfactor_list_pseudo = data["bfactor_list_pseudo"]
    bfactor_list_atomic = data["bfactor_list_atomic"]
    bfactor_correlation_emdb_spearman = data["bfactor_correlation_emdb"]
    EMDB_PDB_ids_present = data["EMDB_PDB_ids_present"]

    filtered_bfactor_list_pseudo = {}
    filtered_bfactor_list_atomic = {}
    filtered_spearman_dict = {}
    filtered_emdb_pdbs = []

    # Filter data based on B-factors
    for emdb_pdb in tqdm(EMDB_PDB_ids_present):
        # Get B-factor lists
        if emdb_pdb in ["0408_6nbd"]:
            print(f"Skipping {emdb_pdb} due to known issues with B-factors")
            continue
        bfactor_pseudo = bfactor_list_pseudo[emdb_pdb]
        bfactor_atomic = bfactor_list_atomic[emdb_pdb]

        # Check if B-factors are proper
        if check_if_bfactors_proper(bfactor_pseudo) and check_if_bfactors_proper(bfactor_atomic):
            if bfactor_correlation_emdb_spearman[emdb_pdb][0] < 0:
                print(f"Negative correlation for {emdb_pdb}")                
            filtered_bfactor_list_pseudo[emdb_pdb] = bfactor_pseudo
            filtered_bfactor_list_atomic[emdb_pdb] = bfactor_atomic
            filtered_spearman_dict[emdb_pdb] = bfactor_correlation_emdb_spearman[emdb_pdb]
            filtered_emdb_pdbs.append(emdb_pdb)
        else:
            print(f"Filtered out {emdb_pdb} due to improper B-factors")
            continue
        
        
    # Print statistics of how many entries were filtered out
    print(f"Original number of entries: {len(EMDB_PDB_ids_present)}")
    print(f"Filtered number of entries: {len(filtered_emdb_pdbs)}")
    print(f"Number of entries filtered out: {len(EMDB_PDB_ids_present) - len(filtered_emdb_pdbs)}")

    output_dict = {
        "EMDB_PDB_ids_present": filtered_emdb_pdbs,
        "bfactor_list_atomic": filtered_bfactor_list_atomic,
        "bfactor_list_pseudo": filtered_bfactor_list_pseudo,
        "bfactor_correlation_emdb": filtered_spearman_dict
    }
            
    
    # Save filtered data
    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)

    print(f"Saved filtered dictionary to {output_path}")

if __name__ == "__main__":
    main()