# 1) IMPORTS 
import os
import sys
import json
from datetime import datetime
import numpy as np

# Mandatory path setup
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])

# Custom locscale functions
from locscale.include.emmer.pdb.pdb_tools import neighborhood_bfactor_correlation
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist

# 2) Set seed for reproducibility
np.random.seed(42)

# 3) Global variables
figure_number = 2
radius_list = [2, 10]  # only two radii used

# 4) SETUP
def main():
    # Set up the data archive path
    data_archive_path = setup_environment()  # Mandatory function

    # INPUT FOLDER
    input_folder = os.path.join(
        data_archive_path, "raw", "general", f"figure_{figure_number}", "neighborhood_correlation"
    )

    # INPUT FILES
    emmap_path = os.path.join(input_folder, "EMD_3061_unfiltered.mrc")
    atomic_model_path = os.path.join(input_folder, "5a63_shifted_servalcat_refined.pdb")
    pseudomodel_with_restrained_refinement_path = os.path.join(input_folder, "EMD_3061_unfiltered_confidenceMap_gradient_pseudomodel_proper_element_composition.pdb")
    pseudomodel_without_restrained_refinement_path = os.path.join(input_folder, "fdr_soft_gradient_pseudomodel_servalcat_refined.pdb")

    # ASSERT FILES EXIST
    assert_paths_exist(
        emmap_path,
        atomic_model_path,
        pseudomodel_with_restrained_refinement_path,
        pseudomodel_without_restrained_refinement_path,
    )

    # OUTPUT FOLDER AND FILE
    output_json_folder = os.path.join(data_archive_path, "outputs", f"supplementary_{figure_number}", "neighborhood_correlation")
    create_folders_if_they_do_not_exist(output_json_folder)
    json_data_path = os.path.join(output_json_folder, "neighborhood_correlation_data.json")

    # COMPUTE MASK FROM ATOMIC MODEL
    atomic_model_mask_path = atomic_model_path.replace(".pdb", "_atomic_mask.mrc")
    atomic_model_mask = get_atomic_model_mask(emmap_path, atomic_model_path, output_filename=atomic_model_mask_path)

    # FUNCTION TO EXTRACT DATA AT A GIVEN RADIUS INDEX
    def extract_bfactor_pair(result_dict, radius_value):
        value = result_dict[radius_value]  # value = (individual_b, neighborhood_b, correlation)
        return value[0], value[1]

    max_radius = max(radius_list)
    num_steps = max_radius
    # RUN COMPUTATION
    bfactor_atomic = neighborhood_bfactor_correlation(
        atomic_model_path, max_radius=max_radius, num_steps=num_steps
    )
    bfactor_restrained = neighborhood_bfactor_correlation(
        pseudomodel_with_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps
    )
    bfactor_restrained_ordered = neighborhood_bfactor_correlation(
        pseudomodel_with_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps, 
        mask_path=atomic_model_mask_path, invert=False
    )
    bfactor_restrained_disordered = neighborhood_bfactor_correlation(
        pseudomodel_with_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps,
        mask_path=atomic_model_mask_path, invert=True
    )

    bfactor_unrestrained = neighborhood_bfactor_correlation(
        pseudomodel_without_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps
    )
    bfactor_unrestrained_ordered = neighborhood_bfactor_correlation(
        pseudomodel_without_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps,
        mask_path=atomic_model_mask_path, invert=False
    )
    bfactor_unrestrained_disordered = neighborhood_bfactor_correlation(
        pseudomodel_without_restrained_refinement_path, max_radius=max_radius, num_steps=num_steps,
        mask_path=atomic_model_mask_path, invert=True
    )



    # STRUCTURE THE DATA
    neighborhood_correlation_data = {}
    for i, radius in enumerate(radius_list):
        individual_bfactors_atomic, neighborhood_bfactors_atomic = extract_bfactor_pair(bfactor_atomic, radius)
        individual_bfactors_restrained, neighborhood_bfactors_restrained = extract_bfactor_pair(bfactor_restrained, radius)
        individual_bfactors_restrained_ordered, neighborhood_bfactors_restrained_ordered = extract_bfactor_pair(bfactor_restrained_ordered, radius)
        individual_bfactors_restrained_disordered, neighborhood_bfactors_restrained_disordered = extract_bfactor_pair(bfactor_restrained_disordered, radius)
        individual_bfactors_unrestrained, neighborhood_bfactors_unrestrained = extract_bfactor_pair(bfactor_unrestrained, radius)
        individual_bfactors_unrestrained_ordered, neighborhood_bfactors_unrestrained_ordered = extract_bfactor_pair(bfactor_unrestrained_ordered, radius)
        individual_bfactors_unrestrained_disordered, neighborhood_bfactors_unrestrained_disordered = extract_bfactor_pair(bfactor_unrestrained_disordered, radius)

        neighborhood_correlation_data[f"radius_{int(radius)}"] = {
            "atomic": {
                "individual_bfactors": individual_bfactors_atomic,
                "neighborhood_bfactors": neighborhood_bfactors_atomic
            },
            "restrained_pseudomodel": {
                "individual_bfactors": individual_bfactors_restrained,
                "neighborhood_bfactors": neighborhood_bfactors_restrained
            },
            "unrestrained_pseudomodel": {
                "individual_bfactors": individual_bfactors_unrestrained,
                "neighborhood_bfactors": neighborhood_bfactors_unrestrained
            },
            "restrained_pseudomodel_ordered": {
                "individual_bfactors": individual_bfactors_restrained_ordered,
                "neighborhood_bfactors": neighborhood_bfactors_restrained_ordered
            },
            "restrained_pseudomodel_disordered": {
                "individual_bfactors": individual_bfactors_restrained_disordered,
                "neighborhood_bfactors": neighborhood_bfactors_restrained_disordered
            },
            "unrestrained_pseudomodel_ordered": {
                "individual_bfactors": individual_bfactors_unrestrained_ordered,
                "neighborhood_bfactors": neighborhood_bfactors_unrestrained_ordered
            },
            "unrestrained_pseudomodel_disordered": {
                "individual_bfactors": individual_bfactors_unrestrained_disordered,
                "neighborhood_bfactors": neighborhood_bfactors_unrestrained_disordered
            }
        }

    # SAVE OUTPUT
    with open(json_data_path, "w") as json_file:
        json.dump(neighborhood_correlation_data, json_file, indent=4)


# SCRIPT ENTRYPOINT
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {end_time - start_time}")
    print("=" * 80)