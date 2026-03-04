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
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from s2a_utils import get_atomic_bfactor_correlation, get_input_files_for_adp_correlation_atomicmodel_pseudomodel

from locscale.preprocessing.headers import run_servalcat_iterative
from locscale.include.emmer.ndimage.map_utils import load_map

np.random.seed(42)
random.seed(42)
# PARAMETERS

def main():
    data_archive_path = setup_environment()
    input_folder = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "bfactor_comparison_pseudo_atomic_3061")
    
    create_folders_if_they_do_not_exist(input_folder)
    
    ## Get input file paths from the dataset
    
    input_files_MF = {}
    input_files_MB = {}
    emdb_pdbs_present = []
    input_files_MF["3061_5a63"] = {
        "halfmap_1_path": os.path.join(input_folder, "model_free", "EMD-3061-half-1.map"),
        "halfmap_2_path": os.path.join(input_folder, "model_free", "EMD-3061-half-2.map"),
        "target_pdb_path": os.path.join(input_folder, "model_free", "fdr_soft_gradient_pseudomodel_uniform_biso.pdb"),
        "resolution": 2.8,
    }
    input_files_MB["3061_5a63"] = {
        "halfmap_1_path": os.path.join(input_folder, "model_based", "EMD-3061-half-1.map"),
        "halfmap_2_path": os.path.join(input_folder, "model_based", "EMD-3061-half-2.map"),
        "target_pdb_path": os.path.join(input_folder, "model_based", "5a63_shifted_uniform_biso.cif"),
        "resolution": 2.8,
    }

    # Run the refinement for model free and model based
    emdb_pdb = "3061_5a63"
    _ = run_servalcat_iterative(
            model_path = input_files_MB[emdb_pdb]["target_pdb_path"],
            map_path = [\
                input_files_MB[emdb_pdb]["halfmap_1_path"],\
                input_files_MB[emdb_pdb]["halfmap_2_path"]\
            ],            
            resolution = input_files_MB[emdb_pdb]["resolution"],
            num_iter = 10, 
            pseudomodel_refinement = False, 
            verbose = True,
    )
    _ = run_servalcat_iterative(
            model_path = input_files_MF[emdb_pdb]["target_pdb_path"],
            map_path = [\
                input_files_MF[emdb_pdb]["halfmap_1_path"],\
                input_files_MF[emdb_pdb]["halfmap_2_path"]\
            ],
            resolution = input_files_MF[emdb_pdb]["resolution"],
            num_iter = 10, 
            pseudomodel_refinement = True, 
            verbose = True,
    )

if __name__ == "__main__":
    main()