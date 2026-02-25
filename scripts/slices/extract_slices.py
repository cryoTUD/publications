## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pandas as pd
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.chapter_3_find_micelle import find_number_of_membranes, \
    extract_dummy_residues_from_pdb, get_membrane, find_best_plane, find_membrane_end, best_fit_plane_section
import math 
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams
from scipy.ndimage import sobel, uniform_filter
from tqdm import tqdm

# LocScale imports
from locscale.include.emmer.ndimage.map_utils import load_map, convert_pdb_to_mrc_position


# Set the seed for reproducibility
np.random.seed(42)

# Global variables
emdb_id = "0257"
pdb_id = "6hra"

def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    input_folder = os.path.join(data_archive_path, "inputs", "3_surfer", "training_data")
    output_folder = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "training_data_analysis")
    # Get input 
    unmodelled_region = os.path.join(input_folder, f"emd_{emdb_id}_difference_mask.mrc")
    input_pdb_path = os.path.join(input_folder, f"{pdb_id}.pdb")

    assert_paths_exist(input_folder, unmodelled_region, input_pdb_path)
    create_folders_if_they_do_not_exist(output_folder)

    output_filename = os.path.join(output_folder, f"slices_extracted_{emdb_id}.pickle")
    
    # Load the unmodelled region
    difference_mask, apix = load_map(unmodelled_region)
    
    # Extract dummy residues 
    N_coord, O_coord = extract_dummy_residues_from_pdb(input_pdb_path)
    num_membranes = find_number_of_membranes(input_pdb_path)

    imsize = len(difference_mask)

    # Get the membrane
    coordinatesN = np.array(convert_pdb_to_mrc_position(N_coord, apix))
    coordinatesO = np.array(convert_pdb_to_mrc_position(O_coord, apix))

    x, y, z = np.ogrid[0:imsize, 0:imsize, 0:imsize]

    planeN, planeO, _ = find_best_plane(coordinatesN, coordinatesO)


    # Load the unmodelled region
    difference_mask, apix = load_map(unmodelled_region)

    # Extract dummy residues 
    N_coord, O_coord = extract_dummy_residues_from_pdb(input_pdb_path)
    num_membranes = find_number_of_membranes(input_pdb_path)

    imsize = len(difference_mask)

    # Get the membrane
    coordinatesN = np.array(convert_pdb_to_mrc_position(N_coord, apix))
    coordinatesO = np.array(convert_pdb_to_mrc_position(O_coord, apix))

    x, y, z = np.ogrid[0:imsize, 0:imsize, 0:imsize]

    print(x.shape, y.shape, z.shape)

    planeN, planeO, _ = find_best_plane(coordinatesN, coordinatesO)

    print(planeN, planeO)
    average_plane = (planeN + planeO) / 2

    a,b,c,d = average_plane
    ax_by_cz = a*x + b*y + c*z 
    print(ax_by_cz.shape)

    
    mem_length = abs(planeN[-1]-planeO[-1])
    laps = math.ceil(a*imsize + b*imsize + c*imsize)
    print(mem_length, laps)

    start = int(max(0, min(planeN[-1] - mem_length, planeO[-1] - mem_length)))
    stop  = int(min(laps, max(planeN[-1] + mem_length, planeO[-1] + mem_length)))

    scan_range = np.arange(start, stop, 1)

    smoothened_mask = uniform_filter(difference_mask, size=5)
    sobel_x = sobel(smoothened_mask, axis=0)
    sobel_y = sobel(smoothened_mask, axis=1)
    sobel_z = sobel(smoothened_mask, axis=2)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    pixels_raw = np.zeros(len(scan_range))
    pixels_smooth = np.zeros(len(scan_range))
    pixels_edge = np.zeros(len(scan_range))
    
    slices_raw = []
    slices_smooth = []
    slices_edge = []

    num_sample_coordinates = 10 
    for i, d in enumerate(tqdm(scan_range)):
        mem_slice = ((ax_by_cz >= d-0.5) & (ax_by_cz <= d+0.5)).astype(np.float32)
        mem_slice_coordinates = np.array(np.where(mem_slice == 1)).T 

        random_coordinate_indices = np.random.choice(mem_slice_coordinates.shape[0], num_sample_coordinates, replace=False)
        random_coordinates = mem_slice_coordinates[random_coordinate_indices]
        difference_mask_slice = best_fit_plane_section(difference_mask, random_coordinates, spacing=1)[0]
        smoothened_mask_slice = best_fit_plane_section(smoothened_mask, random_coordinates, spacing=1)[0]
        sobel_combined_slice = best_fit_plane_section(sobel_combined, random_coordinates, spacing=1)[0]


        pixels_raw[i] = np.sum(mem_slice*difference_mask)
        pixels_smooth[i] = np.sum(mem_slice*smoothened_mask)
        pixels_edge[i] = np.sum(mem_slice*sobel_combined)

        slices_raw.append(difference_mask_slice)
        slices_smooth.append(smoothened_mask_slice)
        slices_edge.append(sobel_combined_slice)

    slices_extracted = {
        "slices_raw": slices_raw,
        "slices_smooth": slices_smooth,
        "slices_edge": slices_edge,
        "pixels_raw": pixels_raw,
        "pixels_smooth": pixels_smooth,
        "pixels_edge": pixels_edge,
        "scan_range": scan_range,
        "emdb_id": emdb_id,
        "pdb_id": pdb_id
    }

    pd.to_pickle(slices_extracted, output_filename)



if __name__ == "__main__":
    main()
