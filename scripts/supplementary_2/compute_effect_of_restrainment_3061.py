# analysis_fsc_iterations.py

## IMPORTS
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])

import numpy as np
import random
import json
import pandas as pd
from tqdm import tqdm
import joblib 

# Custom imports from your project
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from s2_utils import compute_fsc_cycle, jsonify_dictionary
# Set the seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
refmac_iterations = np.arange(1, 51)
def main():
    """
    Main entry point for computing the FSC curves over 50 iterations.
    This script:
    1) Loads the necessary input data (e.g., half-maps, masks, or any raw data).
    2) Performs the FSC calculations in a loop.
    3) Produces a structured output (JSON or CSV) for subsequent plotting.
    """
    # ------------------------------------------------------------------------------
    # 1) ENVIRONMENT SETUP AND PATHS
    # ------------------------------------------------------------------------------
    data_archive_path = setup_environment()  # Mandatory environment setup
    
    # Define input folders/files
    input_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_2", "effect_of_restrainment")
    with_restraints = os.path.join(input_folder_main, "with_restraints")
    without_restraints = os.path.join(input_folder_main, "without_restraints")
    halfmap1_path = os.path.join(without_restraints, "EMD-3061-half-1.map")
    halfmap2_path = os.path.join(without_restraints, "EMD-3061-half-2.map")
    model_map_save_loc_without_restraints = os.path.join(without_restraints, "model_maps")
    model_map_save_loc_with_restraints = os.path.join(with_restraints, "model_maps")
    # Define output folders
    analysis_output_folder = os.path.join(data_archive_path, "processed", "structured_data", "supplementary_2")
    output_filename = os.path.join(analysis_output_folder, "fsc_curves_with_and_without_averaging.json")
    refined_model_per_iteration_without_averaging = {\
        k : os.path.join(without_restraints, "refmac_iteration_50", f"servalcat_refinement_cycle_{k}_no_average.cif")\
        for k in refmac_iterations}

    model_map_paths_without_averaging = {\
        k : os.path.join(model_map_save_loc_without_restraints, f"modelmap_cycle_{k}_before_averaging.mrc")\
        for k in refmac_iterations}
    
    model_map_paths_with_averaging = {\
        k : os.path.join(model_map_save_loc_with_restraints, f"modelmap_cycle_{k}_before_averaging.mrc")\
        for k in refmac_iterations}


    assert_paths_exist(input_folder_main, halfmap1_path, halfmap2_path, model_map_save_loc_without_restraints, model_map_save_loc_with_restraints)
    create_folders_if_they_do_not_exist(analysis_output_folder)

    num_iterations = 50
    fsc_results = []  # We'll store our iteration results in a list of dicts

    # ------------------------------------------------------------------------------
    n_jobs = 10
    verbose = 10
    
    results_without_averaging = joblib.Parallel(\
                                    n_jobs=n_jobs, verbose=10)(\
                                    joblib.delayed(compute_fsc_cycle)\
                                    (cycle, halfmap1_path, halfmap2_path, model_map_paths_without_averaging[cycle])\
                                    for cycle in refmac_iterations)

    results_with_averaging = joblib.Parallel(\
                                    n_jobs=n_jobs, verbose=10)(\
                                    joblib.delayed(compute_fsc_cycle)\
                                    (cycle, halfmap1_path, halfmap2_path, model_map_paths_with_averaging[cycle])\
                                    for cycle in refmac_iterations)
    # ------------------------------------------------------------------------------
    # 4) SAVE RESULTS
    # ------------------------------------------------------------------------------
    fsc_cycles_halfmap1_without_averaging = {}
    fsc_cycles_halfmap2_without_averaging = {}
    for result in results_without_averaging:
        cycle = result["cycle"]
        fsc_cycles_halfmap1_without_averaging[cycle] = result["halfmap1"]
        fsc_cycles_halfmap2_without_averaging[cycle] = result["halfmap2"]

    fsc_cycles_halfmap1_with_averaging = {}
    fsc_cycles_halfmap2_with_averaging = {}
    for result in results_with_averaging:
        cycle = result["cycle"]
        fsc_cycles_halfmap1_with_averaging[cycle] = result["halfmap1"]
        fsc_cycles_halfmap2_with_averaging[cycle] = result["halfmap2"]
    

    results = {
        "fsc_cycles_halfmap1_without_averaging" : fsc_cycles_halfmap1_without_averaging,
        "fsc_cycles_halfmap2_without_averaging" : fsc_cycles_halfmap2_without_averaging,
        "fsc_cycles_halfmap1_with_averaging" : fsc_cycles_halfmap1_with_averaging,
        "fsc_cycles_halfmap2_with_averaging" : fsc_cycles_halfmap2_with_averaging,
        "cycles" : refmac_iterations,
    }

    with open(output_filename, "w") as f:
        json.dump(jsonify_dictionary(results), f, indent=4)


if __name__ == "__main__":
    main()
