
import os 
import numpy as np 
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.include.emmer.ndimage.map_tools import apply_radial_profile
from locscale.emmernet.emmernet_functions import calibrate_variance
from scipy.stats import norm
import seaborn as sns
from locscale.include.emmer.ndimage.map_utils import extract_window
from locscale.emmernet.run_emmernet import preprocess_map 
import random
random.seed(42)
import sys 
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import GroupNormalization

unsharpened_map_path = "/home/abharadwaj1/papers/elife_paper/figure_information/outputs/feature_enhance_test_maps_hybrid_60k/0282_6huo/EMD_282_unsharpened_fullmap.mrc"
pvddt_map_path = "/home/abharadwaj1/papers/elife_paper/figure_information/outputs/feature_enhance_test_maps_hybrid_60k/0282_6huo/pvddt_map.mrc"
emmernet_model_path = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/raw/general/supplementary_5/EMmerNet_highContext.hdf5"
fe_map_path = "/home/abharadwaj1/papers/elife_paper/figure_information/outputs/feature_enhance_test_maps_hybrid_60k/0282_6huo/emd_0282_emmernet_output_mean.mrc"

emmap, apix = load_map(unsharpened_map_path)
pvddt_map, _ = load_map(pvddt_map_path)
fe_map = load_map(fe_map_path)[0]

test_pvddt_point = int(sys.argv[1]) if len(sys.argv) > 1 else 0
epsilon = 0.1
# preprocess the maps 
emmap_preprocessed = preprocess_map(emmap, apix, standardize=True)
fe_map_preprocessed = preprocess_map(fe_map, apix, standardize=False)
pvddt_resampled = preprocess_map(pvddt_map, apix, standardize=False)

fe_mask = fe_map_preprocessed > 0.05 #& (fe_map_preprocessed < 0.1)
voxel_indices = np.where(fe_mask)
print(f"Found {len(voxel_indices[0])}")
all_voxel_indices = [(voxel_indices[0][i], voxel_indices[1][i], voxel_indices[2][i]) for i in range(len(voxel_indices[0]))]
# randomly select 1000 central voxels
random_selected_indices = random.sample(all_voxel_indices, 1000)
pvddt_results = {}
for central_voxel_test in tqdm(random_selected_indices, desc="Processing central voxels"):
    input_cube_emmap = extract_window(emmap_preprocessed, central_voxel_test, size=32)

    model = load_model(emmernet_model_path, custom_objects={'GroupNormalization': GroupNormalization})
    nx, ny, nz = input_cube_emmap.shape
    X_cube = np.zeros((1, nx, ny, nz, 1), dtype=np.float32) 
    X_cube[0, :, :, :, 0] = input_cube_emmap

    X_cube_1_tf = tf.convert_to_tensor(X_cube, dtype=tf.float32)

    cx, cy, cz = 16, 16, 16  # center voxel coordinates in the cube

    monte_carlo_cubes =  [model(X_cube_1_tf, training=True) for i in range(15)]
    monte_carlo_cubes = np.array([cube.numpy().reshape(32,32,32) for cube in monte_carlo_cubes])
    intensity_value_at_center_monte_carlo = [monte_carlo_cubes[i][cz,cy,cx] for i in range(15)]

    scaled_maps_pvddt = [apply_radial_profile(input_cube_emmap, monte_carlo_cubes[i]) for i in range(len(monte_carlo_cubes))]
    central_voxel_intensities_from_scaled_maps_pvddt = [scaled_maps_pvddt[i][cz, cy, cx] for i in range(len(scaled_maps_pvddt))]

    scaled_map_using_mean_pvddt = apply_radial_profile(input_cube_emmap, np.mean(monte_carlo_cubes, axis=0))
    central_voxel_intensity_scaled_map_mean_pvddt = scaled_map_using_mean_pvddt[cz, cy, cx]

    mean_map_pvddt = np.mean(monte_carlo_cubes, axis=0)
    std_dev_map_pvddt = np.std(monte_carlo_cubes, axis=0)
    standard_error_map_pvddt = std_dev_map_pvddt / np.sqrt(len(monte_carlo_cubes))
    calibrated_standard_error_map_pvddt = calibrate_variance(standard_error_map_pvddt)
    calibrated_standard_deviation = calibrated_standard_error_map_pvddt * np.sqrt(len(monte_carlo_cubes))

    mean_map_intensity_central_voxel_pvddt = mean_map_pvddt[cz, cy, cx]
    uncertainty_map_intensity_central_voxel_pvddt = calibrated_standard_error_map_pvddt[cz, cy, cx]

    std_of_central_voxel_intensities_from_scaled_maps_pvddt = np.std(central_voxel_intensities_from_scaled_maps_pvddt)
    mean_central_voxel_intensities_from_scaled_maps_pvddt = np.mean(central_voxel_intensities_from_scaled_maps_pvddt)
    # add secondary y-axis for the calibrated uncertainty

    z_score = (mean_map_intensity_central_voxel_pvddt - central_voxel_intensity_scaled_map_mean_pvddt) / uncertainty_map_intensity_central_voxel_pvddt
    cdf_val = norm.cdf(z_score)
    # rescale to -100 and 100
    rescaled_cdf_val = cdf_val * 200 - 100

    pvddt_results[f"{central_voxel_test[0]}_{central_voxel_test[1]}_{central_voxel_test[2]}"] = {
        "central_voxel": [central_voxel_test[0], central_voxel_test[1], central_voxel_test[2]],
        "mean_intensity_scaled_map": float(mean_central_voxel_intensities_from_scaled_maps_pvddt),
        "std_dev_intensity_scaled_map": float(std_of_central_voxel_intensities_from_scaled_maps_pvddt),
        "rescaled_cdf_value": float(rescaled_cdf_val),
        "mean_map_intensity": float(mean_map_intensity_central_voxel_pvddt),
        "uncertainty_map_intensity": float(uncertainty_map_intensity_central_voxel_pvddt),
        "central_voxel_intensities" : central_voxel_intensities_from_scaled_maps_pvddt,
    }

# collect all std_of_central_voxel_intensities_from_scaled_maps_pvddt
std_of_central_voxel_intensities_from_scaled_maps_pvddt_list = [result["std_dev_intensity_scaled_map"] for result in pvddt_results.values()]
variation_of_CVI_array = np.array(std_of_central_voxel_intensities_from_scaled_maps_pvddt_list)
average_variation_of_CVI = np.mean(std_of_central_voxel_intensities_from_scaled_maps_pvddt_list)
# collect all mean_central_voxel_intensities_from_scaled_maps_pvddt
mean_central_voxel_intensities_from_scaled_maps_pvddt_list = [result["mean_intensity_scaled_map"] for result in pvddt_results.values()]
mean_of_mean_of_CVI = np.mean(mean_central_voxel_intensities_from_scaled_maps_pvddt_list)
# collect all rescaled_cdf_value
rescaled_cdf_values = [result["rescaled_cdf_value"] for result in pvddt_results.values()]
mean_observed_pvddt = np.mean(rescaled_cdf_values)

print(f"Test PVDDT point: {test_pvddt_point}")
print(f"Mean observed pVDDT: {mean_observed_pvddt}")
print(f"Average variation of CVI: {average_variation_of_CVI}")
print(f"Mean of mean of CVI: {mean_of_mean_of_CVI}")


# Save the results to a pandas DataFrame and then to a csv 
import pandas as pd
output_file = f"pvddt_results_test_random_collection.csv"
output_path = os.path.join(os.getcwd(), output_file)
df = pd.DataFrame.from_dict(pvddt_results, orient='index')
df.to_csv(output_path, index=False)



