# %% IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import random
import numpy as np
from datetime import datetime
import pickle
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
from locscale.preprocessing.headers import run_servalcat_iterative
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
from locscale.include.emmer.ndimage.map_utils import save_as_mrc, load_map
from locscale.include.emmer.ndimage.filter import get_cosine_mask

from scripts.utils.general import setup_environment, assert_paths_exist, copy_file_to_folder, create_folders_if_they_do_not_exist
from scripts.figure_2.figure_2_functions import compute_fsc_all_cycles, save_results, refine_model, simulate_maps
# Add custom path (mandatory)

# %% SEED FOR REPRODUCIBILITY
random.seed(42)
np.random.seed(42)

# %% GLOBAL VARIABLES

NUM_ITER = 10
RESOLUTION = 3.4
FIGURE_NUMBER = 2  # Update this if processing a different figure
suffix = ""
# %% MAIN FUNCTION
def main():
    # SETUP ENVIRONMENT

    data_archive_path = setup_environment()  # Set up environment

    input_folder = os.path.join(data_archive_path, "raw", "general", f"figure_{FIGURE_NUMBER}", f"bfactor_comparison_pseudo_atomic{suffix}")
    output_folder = os.path.join(data_archive_path, "processed", "pdbs", f"figure_{FIGURE_NUMBER}", f"bfactor_comparison_pseudo_atomic{suffix}")
    output_folder_to_store = os.path.join(data_archive_path, "processed", "structured_data", f"figure_{FIGURE_NUMBER}", f"bfactor_comparison_pseudo_atomic{suffix}")
    output_folder_pseudomodel_refinement = os.path.join(output_folder, "pseudomodel_refinement")
    output_folder_atomic_refinement = os.path.join(output_folder, "atomic_refinement")
    # CREATE OUTPUT FOLDER
    create_folders_if_they_do_not_exist(output_folder, output_folder_to_store, output_folder_pseudomodel_refinement, output_folder_atomic_refinement)

    hybrid_model_path = os.path.join(input_folder, "cropped_model_pdb_5vkq_integrated_pseudoatoms.cif")
    emmap_path = os.path.join(input_folder, "emd_8702_unsharpened_map.mrc")
    atomic_model_path = os.path.join(input_folder, "PDB_5vkq_unrefined_shifted_servalcat_refined.pdb")

    # model_path = os.path.join(input_folder, "5a63.pdb")
    # halfmap1_path = os.path.join(input_folder, "EMD-3061-half-1.map")
    # halfmap2_path = os.path.join(input_folder, "EMD-3061-half-2.map")
    # confidence_mask_path = os.path.join(input_folder, "EMD_3061_unfiltered_confidenceMap.mrc")

    # ASSERT INPUT FILES EXIST
    assert_paths_exist(hybrid_model_path, emmap_path, atomic_model_path)

    # LOAD DATA    
    
    # COPY FILES TO OUTPUT FOLDER
    for path in [hybrid_model_path, emmap_path]:
        copy_file_to_folder(path, output_folder_pseudomodel_refinement)
    
    for path in [atomic_model_path, emmap_path]:
        copy_file_to_folder(path, output_folder_atomic_refinement)

    print("Refining hybrid model...")
    refined_model_path = refine_model(
        model_path=os.path.join(output_folder_pseudomodel_refinement, os.path.basename(hybrid_model_path)),
        map_path=os.path.join(output_folder_pseudomodel_refinement, os.path.basename(emmap_path)),
        resolution=RESOLUTION,
        num_iter=NUM_ITER,
        pseudomodel_refinement=True,
    )

    print("Refining atomic model...")
    refined_atomic_model_path = refine_model(
        model_path=os.path.join(output_folder_atomic_refinement, os.path.basename(atomic_model_path)),
        map_path=os.path.join(output_folder_atomic_refinement, os.path.basename(emmap_path)),
        resolution=RESOLUTION,
        num_iter=NUM_ITER,
        pseudomodel_refinement=False,
    )

    print(f"Refined model saved to {refined_model_path}")
    print(f"Refined atomic model saved to {refined_atomic_model_path}")



# %% RUN SCRIPT
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {processing_time}")
    print("=" * 80)
