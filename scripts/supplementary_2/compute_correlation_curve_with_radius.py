## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import pandas as pd
import random 

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from locscale.include.emmer.pdb.pdb_tools import neighborhood_bfactor_correlation
from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

def main():
    # Setup environment and define paths
    data_archive_path = setup_environment()
    input_folder = os.path.join(data_archive_path, "raw", "general", "figure_2", "neighborhood_correlation")
    output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2", "neighborhood_correlation")
    correlation_json_path = os.path.join(output_folder, "neighborhood_radius_vs_correlation_values.json")

    assert_paths_exist(
        os.path.join(input_folder, "EMD_3061_unfiltered.mrc"),
        os.path.join(input_folder, "5a63_shifted_servalcat_refined.pdb"),
        os.path.join(input_folder, "EMD_3061_unfiltered_confidenceMap_gradient_pseudomodel_proper_element_composition.pdb"),
        os.path.join(input_folder, "fdr_soft_gradient_pseudomodel_servalcat_refined.pdb")
    )
    create_folders_if_they_do_not_exist(output_folder)

    # Input files
    emmap_path = os.path.join(input_folder, "EMD_3061_unfiltered.mrc")
    atomic_model_path = os.path.join(input_folder, "5a63_shifted_servalcat_refined.pdb")
    restrained_pseudomodel_path = os.path.join(input_folder, "EMD_3061_unfiltered_confidenceMap_gradient_pseudomodel_proper_element_composition.pdb")
    unrestrained_pseudomodel_path = os.path.join(input_folder, "fdr_soft_gradient_pseudomodel_servalcat_refined.pdb")
    atomic_model_mask_path = atomic_model_path.replace(".pdb", "_atomic_mask.mrc")

    if not os.path.exists(atomic_model_mask_path):
        # Compute mask
        get_atomic_model_mask(emmap_path, atomic_model_path, output_filename=atomic_model_mask_path)

    # Compute correlations
    num_steps = 10
    max_radius = 10
    radii = np.linspace(0, max_radius, num_steps)[1:]  # Skip radius 0

    def extract_correlation_dict(result):
        return [v[2][0] for k, v in result.items() if float(k) in radii]

    

    bfactor_neighborhood_atomic = neighborhood_bfactor_correlation(
        atomic_model_path, max_radius=max_radius, num_steps=num_steps)
    bfactor_neighborhood_restrained = neighborhood_bfactor_correlation(
        restrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps)
    bfactor_neighborhood_unrestrained = neighborhood_bfactor_correlation(
        unrestrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps)
    bfactor_neighborhood_restrained_ordered = neighborhood_bfactor_correlation(
        restrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps, mask_path=atomic_model_mask_path, invert=False)
    bfactor_neighborhood_restrained_disordered = neighborhood_bfactor_correlation(
        restrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps, mask_path=atomic_model_mask_path, invert=True)
    bfactor_neighborhood_unrestrained_ordered = neighborhood_bfactor_correlation(
        unrestrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps, mask_path=atomic_model_mask_path, invert=False)
    bfactor_neighborhood_unrestrained_disordered = neighborhood_bfactor_correlation(
        unrestrained_pseudomodel_path, max_radius=max_radius, num_steps=num_steps, mask_path=atomic_model_mask_path, invert=True)

    # Extract values
    radii = list(bfactor_neighborhood_atomic.keys())
    correlations_atomic = [x[2][0] for x in bfactor_neighborhood_atomic.values()]
    correlations_restrained = [x[2][0] for x in bfactor_neighborhood_restrained.values()]
    correlations_unrestrained = [x[2][0] for x in bfactor_neighborhood_unrestrained.values()]
    correlations_restrained_ordered = [x[2][0] for x in bfactor_neighborhood_restrained_ordered.values()]
    correlations_restrained_disordered = [x[2][0] for x in bfactor_neighborhood_restrained_disordered.values()]
    correlations_unrestrained_ordered = [x[2][0] for x in bfactor_neighborhood_unrestrained_ordered.values()]
    correlations_unrestrained_disordered = [x[2][0] for x in bfactor_neighborhood_unrestrained_disordered.values()]
    
    corr_data = {
        "radii" : radii
    }
    
    corr_data["atomic"] = correlations_atomic
    corr_data["restrained_pseudomodel"] = correlations_restrained
    corr_data["unrestrained_pseudomodel"] = correlations_unrestrained
    corr_data["restrained_pseudomodel_ordered"] = correlations_restrained_ordered
    corr_data["restrained_pseudomodel_disordered"] = correlations_restrained_disordered
    corr_data["unrestrained_pseudomodel_ordered"] = correlations_unrestrained_ordered
    corr_data["unrestrained_pseudomodel_disordered"] = correlations_unrestrained_disordered

    with open(correlation_json_path, 'w') as f:
        json.dump(corr_data, f, indent=4)

    print(f"Saved correlation radius profile to {correlation_json_path}. Proceeding to plotting...")

if __name__ == "__main__":
    main()