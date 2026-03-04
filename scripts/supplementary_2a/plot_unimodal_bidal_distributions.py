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
    output_path = os.path.join(output_folder, "modality_analysis.pdf")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    emdb_pdbs = data["EMDB_PDB_ids_present"]
    bfactor_list_atomic = data["bfactor_list_atomic"]
    bfactor_list_pseudo = data["bfactor_list_pseudo"]
    correlation_dict = data["bfactor_correlation_emdb"]

    

if __name__ == "__main__":
    main()
