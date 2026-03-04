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
REFINE_MODEL = True
SIMULATE_MAPS = True
CALCULATE_FSC = True

NUM_ITER = 50
RESOLUTION = 3.4
FIGURE_NUMBER = 2  # Update this if processing a different figure
suffix = "_3061"
# %% MAIN FUNCTION
def main():
    # SETUP ENVIRONMENT

    data_archive_path = setup_environment()  # Set up environment

    input_folder = os.path.join(data_archive_path, "raw", "general", f"figure_{FIGURE_NUMBER}", f"overfitting_analysis{suffix}")
    output_folder = os.path.join(data_archive_path, "processed", "pdbs", f"figure_{FIGURE_NUMBER}", f"overfitting_analysis{suffix}_unrestrained")
    output_folder_to_store = os.path.join(data_archive_path, "processed", "structured_data", f"figure_{FIGURE_NUMBER}", f"overfitting_analysis{suffix}_unrestrained")

    # CREATE OUTPUT FOLDER
    create_folders_if_they_do_not_exist(output_folder, output_folder_to_store)
    if suffix == "_8702":
        model_path = os.path.join(input_folder, "cropped_model_pdb_5vkq_integrated_pseudoatoms.cif")
        halfmap1_path = os.path.join(input_folder, "emd_8702_half_map_1.map")
        halfmap2_path = os.path.join(input_folder, "emd_8702_half_map_2.map")
        confidence_mask_path = os.path.join(input_folder, "emd_8702_FDR_confidence_final.map")
    elif suffix == "_3061":
        model_path = os.path.join(input_folder, "fdr_soft_gradient_pseudomodel_servalcat_refined.pdb")
        halfmap1_path = os.path.join(input_folder, "EMD-3061-half-1.map")
        halfmap2_path = os.path.join(input_folder, "EMD-3061-half-2.map")
        confidence_mask_path = os.path.join(input_folder, "EMD_3061_unfiltered_confidenceMap.mrc")

    # ASSERT INPUT FILES EXIST
    assert_paths_exist(model_path, halfmap1_path, halfmap2_path, confidence_mask_path)

    # LOAD DATA
    halfmap1, apix = load_map(halfmap1_path)
    halfmap2, _ = load_map(halfmap2_path)
    mask = load_map(confidence_mask_path)[0]
    mask_binarized = (mask >= 0.99).astype(np.int_)
    softmask = get_cosine_mask(mask_binarized, 3)
    
    # COPY FILES TO OUTPUT FOLDER
    for path in [model_path, halfmap1_path, halfmap2_path]:
        copy_file_to_folder(path, output_folder)

    if REFINE_MODEL:
        print("Refining model...")
        refined_model_path = refine_model(
            model_path=os.path.join(output_folder, os.path.basename(model_path)),
            map_path=os.path.join(output_folder, os.path.basename(halfmap1_path)),
            resolution=RESOLUTION,
            num_iter=NUM_ITER
        )

    if SIMULATE_MAPS:
        print("Simulating maps...")
        simulated_maps = simulate_maps(
            apix=apix,
            shape=mask.shape,
            output_folder=output_folder,
            num_iter=NUM_ITER
        )

    if CALCULATE_FSC:
        print("Calculating FSC...")
        results = compute_fsc_all_cycles(
            output_folder=output_folder, 
            halfmap1_path=halfmap1_path,
            halfmap2_path=halfmap2_path,
            mask_path=confidence_mask_path, 
            num_iter=NUM_ITER,
            n_jobs=10)

        # SAVE RESULTS
        save_results(
            results=results, 
            output_folder=output_folder_to_store, 
            num_iter=NUM_ITER)



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
