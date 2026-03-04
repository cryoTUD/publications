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
from scripts.figure_3.figure_3_functions import preprocess_map, load_maps, calculate_phase_correlations
from locscale.include.emmer.ndimage.map_utils import load_map, extract_window
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array, estimate_bfactor_standard
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist, any_files_are_missing
import warnings 
from locscale.include.emmer.pdb.pdb_tools import find_wilson_cutoff
warnings.filterwarnings("ignore")
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

    # Save the results as a JSON file
    output_file = os.path.join(output_structured_data_folder, f"figure_{figure_number}_bfactor_correlations.pickle")

    # Initialize dictionary for bfactor correlations
    bfactors_in_fdr_mask_all_emdb = {}
    bfactors_in_atomic_mask_all_emdb = {}

    for emdb_pdb in tqdm(os.listdir(emmernet_output_dir)):
        emdb, pdb = emdb_pdb.split("_")
        
        unsharpened_map_path = os.path.join(emmernet_output_dir, emdb_pdb, f"EMD_{int(emdb)}_unsharpened_fullmap.mrc")
        emmernet_output_path = os.path.join(emmernet_output_dir, emdb_pdb, f"emd_{emdb}_emmernet_output.mrc")
        target_model_map_path = os.path.join(target_model_map_dir, f"emd_{emdb}_hybrid_model_map_refined_version_C.mrc")
        fdr_mask_path = os.path.join(confidence_mask_dir, f"emd_{emdb}_FDR_confidence_final.map")
        atomic_mask_path = os.path.join(atomic_mask_dir, f"atomic_model_mask_{emdb_pdb}_strict_3A.mrc")
        
        if any_files_are_missing(
            unsharpened_map_path, emmernet_output_path, target_model_map_path, fdr_mask_path, atomic_mask_path
            ):
            continue

        
        unsharpened_map, apix = load_map(unsharpened_map_path)
        fdr_mask = load_map(fdr_mask_path)[0]
        atomic_mask = load_map(atomic_mask_path)[0]
        
        emmernet_map, _ = load_map(emmernet_output_path)
        target_model_map, _ = load_map(target_model_map_path)
        
        voxels_in_fdr_mask = np.asarray(np.where(fdr_mask > 0.99)).T.tolist()
        voxels_in_atomic_mask = np.asarray(np.where(atomic_mask > 0.5)).T.tolist()
        
        # exclude voxels near the edges of the map
        buffer_size =15
        voxel_near_z_edge = lambda voxel: voxel[0] < buffer_size or voxel[0] > fdr_mask.shape[0] - buffer_size 
        voxel_near_y_edge = lambda voxel: voxel[1] < buffer_size or voxel[1] > fdr_mask.shape[1] - buffer_size
        voxel_near_x_edge = lambda voxel: voxel[2] < buffer_size or voxel[2] > fdr_mask.shape[2] - buffer_size
        avoid_this_voxel = lambda voxel: voxel_near_z_edge(voxel) or voxel_near_y_edge(voxel) or voxel_near_x_edge(voxel)
        
        sample_size = 200 
        sample_voxels_in_fdr_mask = random.sample(voxels_in_fdr_mask, sample_size)
        sample_voxels_in_atomic_mask = random.sample(voxels_in_atomic_mask, sample_size)
        
        # exclude voxels near the edges of the map
        sample_voxels_in_fdr_mask_filter = [voxel for voxel in sample_voxels_in_fdr_mask if not avoid_this_voxel(voxel)]
        sample_voxels_in_atomic_mask_filter = [voxel for voxel in sample_voxels_in_atomic_mask if not avoid_this_voxel(voxel)]
        bfactors_in_fdr_mask = []
        bfactors_in_atomic_mask = []
        
        window_size_px = int(round(25/apix))
        
        wilson_cutoff_global = find_wilson_cutoff(mask_path=fdr_mask_path, verbose=False)
        fsc_cutoff = apix * 2 + 1 
        for center in sample_voxels_in_fdr_mask_filter:
            emmernet_window = extract_window(emmernet_map, center, window_size_px)
            target_model_window = extract_window(target_model_map, center, window_size_px)
            
            rp_emmernet = compute_radial_profile(emmernet_window)
            rp_target_model = compute_radial_profile(target_model_window)
            # num_atoms = rp_target_model[0]
            # mol_weight = num_atoms * 16
            # wilson_cutoff_local = 1/(0.309 * np.power(mol_weight, -1/12))   ## From Amit Singer
            # wilson_cutoff_local = np.clip(wilson_cutoff_local, fsc_cutoff * 1.5, wilson_cutoff_global)
            freq = frequency_array(rp_emmernet, apix)
            try:
                bfactor_emmernet, qfit_emmernet = estimate_bfactor_standard(freq, rp_emmernet, wilson_cutoff=wilson_cutoff_global, fsc_cutoff=fsc_cutoff, return_fit_quality=True)
                bfactor_target_model, qfit_target = estimate_bfactor_standard(freq, rp_target_model, wilson_cutoff=wilson_cutoff_global, fsc_cutoff=fsc_cutoff, return_fit_quality=True)
                
                bfactors_in_fdr_mask.append((bfactor_emmernet, bfactor_target_model, qfit_emmernet, qfit_target))
            except:
                continue    
            
        for center in sample_voxels_in_atomic_mask_filter:
            emmernet_window = extract_window(emmernet_map, center, window_size_px)
            target_model_window = extract_window(target_model_map, center, window_size_px)
            
            rp_emmernet = compute_radial_profile(emmernet_window)
            rp_target_model = compute_radial_profile(target_model_window)
            
            freq = frequency_array(rp_emmernet, apix)
            
            try:
                bfactor_emmernet, qfit_emmernet = estimate_bfactor_standard(freq, rp_emmernet, wilson_cutoff=wilson_cutoff_global, fsc_cutoff=fsc_cutoff, return_fit_quality=True)
                bfactor_target_model, qfit_target = estimate_bfactor_standard(freq, rp_target_model, wilson_cutoff=wilson_cutoff_global, fsc_cutoff=fsc_cutoff, return_fit_quality=True)
                
                bfactors_in_atomic_mask.append((bfactor_emmernet, bfactor_target_model, qfit_emmernet, qfit_target))
            except:
                continue
                
        bfactors_in_fdr_mask_all_emdb[emdb_pdb] = bfactors_in_fdr_mask
        bfactors_in_atomic_mask_all_emdb[emdb_pdb] = bfactors_in_atomic_mask
    

    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "bfactors_in_fdr_mask_all_emdb": bfactors_in_fdr_mask_all_emdb,
                "bfactors_in_atomic_mask_all_emdb": bfactors_in_atomic_mask_all_emdb
            },
            f,
        )
    print(f"Results saved to {output_file}")
if __name__ == "__main__":
    main()
