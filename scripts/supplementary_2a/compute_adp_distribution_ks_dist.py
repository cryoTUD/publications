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
from scipy.stats import invgamma

from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
from s2a_utils import check_if_bfactors_proper, invgamma_permutation_test

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "atomic_v_pseudoatomic_adp_correlation_filtered.pickle")
    output_folder = os.path.join(data_archive_path, "structured_data", "supplementary_2a")
    output_path = os.path.join(output_folder, "permutation_test_results_adp_correlations.pickle")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    bfactor_list_pseudo = data["bfactor_list_pseudo"]
    bfactor_list_atomic = data["bfactor_list_atomic"]
    bfactor_correlation_emdb_spearman = data["bfactor_correlation_emdb_spearman"]
    EMDB_PDB_ids_present = data["EMDB_PDB_ids_present"]

    ks_distance_emdb_pseudo = {}
    ks_distance_emdb_atomic = {}
    ks_pvalues_emdb_pseudo = {}
    ks_pvalues_emdb_atomic = {}

    # Filter data based on B-factors
    for emdb_pdb in tqdm(EMDB_PDB_ids_present):
        bfactor_pseudo_embd = bfactor_list_pseudo[emdb_pdb]
        bfactor_atomic_emdb = bfactor_list_atomic[emdb_pdb]

        bfactors_are_proper = check_if_bfactors_proper(bfactor_atomic_emdb) and check_if_bfactors_proper(bfactor_pseudo_embd)
        if not bfactors_are_proper:
            continue
        invgamma_fit_emdb_pseudo = invgamma.fit(bfactor_pseudo_embd)
        invgamma_fit_emdb_atomic = invgamma.fit(bfactor_atomic_emdb)
        ks_test_pseudo = invgamma_permutation_test(bfactor_pseudo_embd, invgamma_fit_emdb_pseudo)
        ks_test_atomic = invgamma_permutation_test(bfactor_atomic_emdb, invgamma_fit_emdb_atomic)

        ks_distance_emdb_pseudo[emdb_pdb] = ks_test_pseudo.statistic
        ks_distance_emdb_atomic[emdb_pdb] = ks_test_atomic.statistic
        ks_pvalues_emdb_pseudo[emdb_pdb] = ks_test_pseudo.pvalue
        ks_pvalues_emdb_atomic[emdb_pdb] = ks_test_atomic.pvalue


    ks_distance_pseudo_values = list(ks_distance_emdb_pseudo.values())
    ks_distance_atomic_values = list(ks_distance_emdb_atomic.values())
    ks_pvalues_pseudo_values = list(ks_pvalues_emdb_pseudo.values())
    ks_pvalues_atomic_values = list(ks_pvalues_emdb_atomic.values())

    ks_distances_pickle_dict = {
        "ks_distance_pseudo_values" : ks_distance_pseudo_values,
        "ks_distance_atomic_values" : ks_distance_atomic_values,
        "ks_pvalues_pseudo_values" : ks_pvalues_pseudo_values,
        "ks_pvalues_atomic_values" : ks_pvalues_atomic_values,
        "EMDB_PDB_ids_present" : EMDB_PDB_ids_present,
        "information" : \
        "The values of KS distance and p-values are found using permutation tests with 10000 resample using the KS statistic"
    }

    with open(output_path, "wb") as f:
        pickle.dump(ks_distances_pickle_dict, f)

    print(f"Saved filtered dictionary to {output_path}")

if __name__ == "__main__":
    main()