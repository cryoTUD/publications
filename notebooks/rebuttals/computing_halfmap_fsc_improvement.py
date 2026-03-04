## IMPORTS 
import os
import sys 
LOCSCALE_2_SCRIPTS_PATH = "/home/abharadwaj1/papers/publications/2025_LocScale-2.0"
PLOT_DATA_STORE_PATH = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/organized/data"
sys.path.append(LOCSCALE_2_SCRIPTS_PATH)
from scripts.utils.plot_utils import *

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import pickle
import pandas as pd
from scripts.utils.plot_utils import pretty_plot_radial_profile, temporary_rcparams, configure_plot_scaling
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, estimate_bfactor_standard, frequency_array
random.seed(42)
np.random.seed(42)

def compute_masked_fsc_between_two_maps(map1_path, map2_path, mask_path):
    from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
    from locscale.include.emmer.ndimage.filter import get_cosine_mask
    from locscale.include.emmer.ndimage.profile_tools import frequency_array

    map1, apix = load_map(map1_path)
    map2, apix = load_map(map2_path)
    mask, _ = load_map(mask_path)

    mask_binarised = (mask > 0.5).astype(int)
    smooth_mask = get_cosine_mask(mask_binarised, 5)

    masked_map_1 = map1 * smooth_mask
    masked_map_2 = map2 * smooth_mask

    fsc_curve = calculate_fsc_maps(masked_map_1, masked_map_2)
    freq_array = frequency_array(fsc_curve, apix=apix)
    return freq_array, fsc_curve

def download_halfmaps(emdb_id):
    python_script = "/home/abharadwaj1/common_scripts/download_emdb.py"
    os.system(f"python3 {python_script} {emdb_id}")
    halfmap_1_path = f"emd_{emdb_id}/emd_{emdb_id}_half_map_1.map"
    halfmap_2_path = f"emd_{emdb_id}/emd_{emdb_id}_half_map_2.map"
    if os.path.exists(halfmap_1_path) and os.path.exists(halfmap_2_path):
        return halfmap_1_path, halfmap_2_path
    else:
        return None

def run_fem_with_one_halfmap(halfmap_1_path, halfmap_2_path, choose=1):
    import subprocess

    locscale_cmd = ["locscale"]
    locscale_cmd.append("feature_enhance")
    locscale_cmd.append("-em")
    if choose == 1:
        locscale_cmd.append(halfmap_1_path)
    else:
        locscale_cmd.append(halfmap_2_path)
    locscale_cmd.append("-v")
    output_filename = halfmap_1_path.replace(".map", f"_fem_halfmap_{choose}.mrc") if choose == 1 else halfmap_2_path.replace(".map", f"_fem_halfmap_{choose}.mrc")
    output_filename = os.path.basename(output_filename)
    locscale_cmd.append("-o")
    locscale_cmd.append(output_filename)
    locscale_cmd.append("-gpus")
    locscale_cmd.append("1")
    locscale_cmd.append("-np")
    locscale_cmd.append("6")

    try: 
        subprocess.run(locscale_cmd, check=True)
        if not os.path.exists(output_filename):
            print("FEM output file not found: ", output_filename)
            return None
        else:
            baseline_map = output_filename.replace(".mrc", "_baseline.mrc")
            return output_filename, baseline_map
    except subprocess.CalledProcessError as e:
        print("Error running LocScale-2 FEM: ", e)
        return None

def create_experiment_from_halfmap_paths(emdb_id):
    import shutil
    halfmap_paths = download_halfmaps(emdb_id)
    if halfmap_paths is None:
        return None
    halfmap_1_path, halfmap_2_path = halfmap_paths

    halfmap_dir = os.path.dirname(halfmap_1_path)

    fem_with_halfmap_1_dir = os.path.join(halfmap_dir, "fem_with_halfmap_1")
    fem_with_halfmap_2_dir = os.path.join(halfmap_dir, "fem_with_halfmap_2")
    
    create_folders_if_they_do_not_exist(fem_with_halfmap_1_dir, fem_with_halfmap_2_dir)
    # copy halfmaps to respective directories
    copied_halfmap_1_path_study_1 = os.path.join(fem_with_halfmap_1_dir, os.path.basename(halfmap_1_path))
    copied_halfmap_2_path_study_1 = os.path.join(fem_with_halfmap_1_dir, os.path.basename(halfmap_2_path))
    copied_halfmap_1_path_study_2 = os.path.join(fem_with_halfmap_2_dir, os.path.basename(halfmap_1_path))
    copied_halfmap_2_path_study_2 = os.path.join(fem_with_halfmap_2_dir, os.path.basename(halfmap_2_path))

    shutil.copy2(halfmap_1_path, copied_halfmap_1_path_study_1)
    shutil.copy2(halfmap_2_path, copied_halfmap_2_path_study_1)
    shutil.copy2(halfmap_1_path, copied_halfmap_1_path_study_2)
    shutil.copy2(halfmap_2_path, copied_halfmap_2_path_study_2)

    print(f"Copied halfmaps to {fem_with_halfmap_1_dir} and {fem_with_halfmap_2_dir}")
    # run fem with halfmap 1
    print("Running FEM with halfmap 1")
    results_from_fem_analysis_1 = run_fem_with_one_halfmap(copied_halfmap_1_path_study_1, copied_halfmap_2_path_study_1, choose=1)
    if results_from_fem_analysis_1 is None:
        print("FEM with halfmap 1 failed")
        return None
    fem_halfmap_1_path, baseline_map_1 = results_from_fem_analysis_1
    fem_halfmap_1_path = os.path.join(fem_with_halfmap_1_dir, os.path.basename(fem_halfmap_1_path))
    baseline_map_1 = os.path.join(fem_with_halfmap_1_dir, os.path.basename(baseline_map_1))


    # run fem with halfmap 2
    print("Running FEM with halfmap 2")
    results_from_fem_analysis_2 = run_fem_with_one_halfmap(copied_halfmap_1_path_study_2, copied_halfmap_2_path_study_2, choose=2)
    if results_from_fem_analysis_2 is None:
        print("FEM with halfmap 2 failed")
        return None
    fem_halfmap_2_path, baseline_map_2 = results_from_fem_analysis_2
    fem_halfmap_2_path =  os.path.join(fem_with_halfmap_2_dir, os.path.basename(fem_halfmap_2_path))
    baseline_map_2 = os.path.join(fem_with_halfmap_2_dir, os.path.basename(baseline_map_2))
    

    if (fem_halfmap_1_path is None) or (fem_halfmap_2_path is None) or (baseline_map_1 is None) or (baseline_map_2 is None):
        return None
    else:
        confidence_mask_paths_halfmap_1 = os.path.join(fem_with_halfmap_1_dir, "processing_files", f"emd_{emdb_id}_half_map_1_confidenceMap.mrc")
        confidence_mask_paths_halfmap_2 = os.path.join(fem_with_halfmap_2_dir, "processing_files", f"emd_{emdb_id}_half_map_2_confidenceMap.mrc")
        all_paths = {
            "halfmap_1_path" : halfmap_1_path,
            "halfmap_2_path" : halfmap_2_path,
            "fem_halfmap_1_path": fem_halfmap_1_path,
            "fem_halfmap_2_path": fem_halfmap_2_path,
            "confidence_mask_path_halfmap_1": confidence_mask_paths_halfmap_1,
            "confidence_mask_path_halfmap_2": confidence_mask_paths_halfmap_2,
            "baseline_map_1": baseline_map_1,
            "baseline_map_2": baseline_map_2
        }
        return all_paths

    


#emdb_pdbs = [0282_6huo  0311_6hz5  0560_6nzu  10365_6t23  20220_6oxl  20226_6p07  3545_5mqf  4141_5m1s  4531_6qdw  4571_6qk7  4997_6rtc  7127_6bpq  8702_5vkq  9610_6adq]
emdb_pdbs = ["0282_6huo", "0311_6hz5", "0560_6nzu", "10365_6t23", "20220_6oxl", "20226_6p07", "3545_5mqf", "4141_5m1s", "4531_6qdw", "4571_6qk7", "4997_6rtc", "7127_6bpq", "8702_5vkq", "9610_6adq"]

emdbs = [emdb_pdb.split("_")[0] for emdb_pdb in emdb_pdbs]

for emdb in emdbs:
    print(f"Processing EMDB: {emdb}")
    results = create_experiment_from_halfmap_paths(emdb)
    if results is None:
        print(f"Skipping EMDB {emdb} due to errors in FEM processing")
        continue
    
    halfmap_1_path = results["halfmap_1_path"]
    halfmap_2_path = results["halfmap_2_path"]
    fem_halfmap_1_path = results["fem_halfmap_1_path"]
    fem_halfmap_2_path = results["fem_halfmap_2_path"]
    confidence_mask_path_halfmap_1 = results["confidence_mask_path_halfmap_1"]
    confidence_mask_path_halfmap_2 = results["confidence_mask_path_halfmap_2"]
    baseline_map_1 = results["baseline_map_1"]
    baseline_map_2 = results["baseline_map_2"]

     # Compute FSC curves

    print("Computing FSC for FEM with halfmap 1")
    freq_array_1, fsc_curve_1 = compute_masked_fsc_between_two_maps(fem_halfmap_1_path, halfmap_2_path, confidence_mask_path_halfmap_1)

    print("Computing FSC for FEM with halfmap 2")
    freq_array_2, fsc_curve_2 = compute_masked_fsc_between_two_maps(halfmap_1_path, fem_halfmap_2_path, confidence_mask_path_halfmap_2)

    print("Compute FSC for Baseline maps")
    freq_array_baseline_1, fsc_curve_baseline_1 = compute_masked_fsc_between_two_maps(baseline_map_1, halfmap_2_path, confidence_mask_path_halfmap_1)
    freq_array_baseline_2, fsc_curve_baseline_2 = compute_masked_fsc_between_two_maps(halfmap_1_path, baseline_map_2, confidence_mask_path_halfmap_2)

    print("Compute control FSC between halfmaps")
    freq_array_control, fsc_curve_control = compute_masked_fsc_between_two_maps(halfmap_1_path, halfmap_2_path, confidence_mask_path_halfmap_1)

    # save FSC curves to a pickle file
    fsc_data = {
        "freq_array_1": freq_array_1,
        "fsc_curve_1": fsc_curve_1,
        "freq_array_2": freq_array_2,
        "fsc_curve_2": fsc_curve_2,
        "freq_array_baseline_1": freq_array_baseline_1,
        "fsc_curve_baseline_1": fsc_curve_baseline_1,
        "freq_array_baseline_2": freq_array_baseline_2,
        "fsc_curve_baseline_2": fsc_curve_baseline_2,
        "freq_array_control": freq_array_control,
        "fsc_curve_control": fsc_curve_control,
        "all_paths" : results
    }
    fsc_pickle_path = f"emd_{emdb}/{emdb}.pkl"
    with open(fsc_pickle_path, "wb") as f:
        pickle.dump(fsc_data, f)
    print(f"FSC data saved to {fsc_pickle_path}")


    # Plotting the FSC curves as two subplots with shared x-axis one above the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax1.plot(freq_array_1, fsc_curve_1, label="FEM Halfmap 1 vs Halfmap 2", color="blue")
    ax1.plot(freq_array_2, fsc_curve_2, label="Halfmap 1 vs FEM Halfmap 2", color="orange")
    ax1.plot(freq_array_control, fsc_curve_control, label="Halfmap 1 vs Halfmap 2 (Control)", color="green", linestyle="--")
    ax1.set_title(f"FSC Curves for EMDB {emdb} - FEM with Halfmaps")
    ax1.set_ylabel("FSC")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.plot(freq_array_baseline_1, fsc_curve_baseline_1, label="Baseline Halfmap 1 vs Halfmap 2", color="red")
    ax2.plot(freq_array_baseline_2, fsc_curve_baseline_2, label="Halfmap 1 vs Baseline Halfmap 2", color="purple")
    ax2.plot(freq_array_control, fsc_curve_control, label="Halfmap 1 vs Halfmap 2 (Control)", color="green", linestyle="--")
    ax2.set_title(f"FSC Curves for EMDB {emdb} - Baseline")
    ax2.set_xlabel("Spatial Frequency (1/Å)")
    ax2.set_ylabel("FSC")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = f"emd_{emdb}/fsc_comparison_emdb_{emdb}.pdf"
    plt.savefig(plot_path)
    print(f"FSC comparison plot saved to {plot_path}")
    plt.close()

