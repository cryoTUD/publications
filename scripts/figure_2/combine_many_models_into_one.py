## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import gemmi 
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)
compute_for = "bfactor_refinement" # or "bfactor_refinement" # or "pseudomodel_refinement"
folder_name = "pseudomodel_structures_iterations" if compute_for == "pseudomodel_refinement" else "overfitting_analysis"
file_pattern = "pseudoatomic_model_{}.mmcif" if compute_for == "pseudomodel_refinement" else "servalcat_refinement_cycle_{}.cif"
output_filename = "pseudomodel_combined_model_8702.pdb" if compute_for == "pseudomodel_refinement" else "bfactor_refined_8702.pdb"
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", folder_name)
    start_index = 1 
    end_index = 50 if compute_for == "pseudomodel_refinement" else 31 
    
    output_folder_main = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "combined_models")
    combined_model_file_path = os.path.join(output_folder_main, output_filename)
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(data_input_folder_main)
    create_folders_if_they_do_not_exist(output_folder_main)

    file_paths = {cycle: os.path.join(data_input_folder_main, file_pattern.format(cycle)) for cycle in range(start_index, end_index)}
    combined_gemmi_structure = gemmi.Structure()

    for cycle in tqdm(range(start_index, end_index), desc="Combining models"):
        # Load the structure
        gemmi_structure = gemmi.read_structure(file_paths[cycle])
        
        # Extract the model
        model = gemmi_structure[0]
        # Add the model to the combined structure
        combined_gemmi_structure.add_model(model, pos=cycle)
        print(f"Number of models in combined structure: {len(combined_gemmi_structure)}")
    
    # Save the combined structure
    #combined_gemmi_structure.make_mmcif_document().write_file(pseudomodel_iteration_combined_model)
    combined_gemmi_structure.write_pdb(combined_model_file_path)
    print(f"Data saved to {combined_model_file_path}. Please check.")

if __name__ == "__main__":
    main()

