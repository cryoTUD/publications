## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import pickle
from joblib import Parallel, delayed
# Custom imports
from locscale.include.emmer.ndimage.map_utils import load_map
from scripts.figure_3.figure_3_functions import preprocess_map, load_maps, calculate_phase_correlations
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist

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
    unsharp_maps_folder = os.path.join(input_maps_folder, "unsharp_maps")
    target_maps_folder = os.path.join(input_maps_folder, "target_maps")
    mask_folder = os.path.join(input_maps_folder, "mask")

    output_structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data", f"figure_{figure_number}")

    assert_paths_exist(
        input_maps_folder, unsharp_maps_folder, target_maps_folder, mask_folder,
    )

    create_folders_if_they_do_not_exist(
        output_structured_data_folder
    )

    # Save the results as a JSON file
    output_file = os.path.join(output_structured_data_folder, f"figure_{figure_number}_phase_correlations.pickle")

    # Initialize dictionary for phase correlations
    phase_correlations_data = {} 

    print(f"Computing phase and amplitude correlations for:")
    print(f"{emdb_ids}")
    # Loop over each EMDB ID to compute phase correlations
    for emdb_id in tqdm(emdb_ids, desc="Processing EMDB IDs"):
        unsharp_map_path = os.path.join(unsharp_maps_folder, input_map_format.format(int(emdb_id)))
        target_map_path = os.path.join(target_maps_folder, target_map_format.format(emdb_id))
        mask_path = os.path.join(mask_folder, mask_format.format(emdb_id))


        # Compute phase correlations
        phase_correlations_data[emdb_id] = calculate_phase_correlations(unsharp_map_path, target_map_path, mask_path, num_samples=100)

    # Dump the dictionary to a pickle file 
    with open(output_file, "wb") as f:
        pickle.dump(phase_correlations_data, f)
        
    print(f"Phase and amplitude correlations data saved to {output_file}")

if __name__ == "__main__":
    main()
