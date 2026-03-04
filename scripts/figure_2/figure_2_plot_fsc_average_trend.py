## IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
from scripts.utils.plot_utils import *
import numpy as np
import gemmi
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import random 
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import pickle

# Custom imports
from locscale.preprocessing.pseudomodel_classes import extract_model_from_mask
from locscale.include.emmer.ndimage.map_utils import load_map, measure_mask_parameters, save_as_mrc
from locscale.include.emmer.ndimage.filter import get_cosine_mask
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region, estimate_global_bfactor_map_standard
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps, calculate_amplitude_correlation_maps
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array

# Import helper functions 
from figure_2_functions import gradient_solver, create_modmap
from scripts.utils.plot_utils import pretty_lineplot_XY, temporary_rcparams, configure_plot_scaling
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist
# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set global variables
SAVE_FIGURE = True
figure_number = 2
## SETUP 
def main():
    data_archive_path = setup_environment()

    # DEFINE THE PATHS 
    #input_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/pseudomodel_during_iterations/hybrid_pseudomodel_iterations"
    input_maps_folder = os.path.join(data_archive_path, "raw","maps")
    output_maps_folder = os.path.join(data_archive_path, "processed", "maps")
    input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")
    output_pdbs_folder = os.path.join(data_archive_path, "processed", "pdbs")
    output_general_folder = os.path.join(data_archive_path, "processed", "general")
    output_structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data")
    figure_output_folder = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}")
 
    output_pseudomodel_structure_folder = os.path.join(output_pdbs_folder, f"figure_{figure_number}", "pseudomodel_structures_iterations")
    output_pseudomodel_map_folder = os.path.join(output_maps_folder, f"figure_{figure_number}", "pseudomodel_maps")
    output_folder_for_data = os.path.join(output_structured_data_folder, f"figure_{figure_number}")
    output_processing_folder = os.path.join(output_general_folder, f"figure_{figure_number}_processed")
    
    create_folders_if_they_do_not_exist(\
        figure_output_folder,\
    )

    # Output paths
    path_to_store_fsc_average_pickle = os.path.join(output_folder_for_data, "fsc_average_iterations.pickle")
    path_to_store_radial_profile_pickle = os.path.join(output_folder_for_data, "radial_profile_iterations.pickle")

    # Figure output paths
    fsc_average_figure_path = os.path.join(figure_output_folder, f"figure_{figure_number}_fsc_average_vs_iteration.pdf")
    fsc_average_figure_path_png = os.path.join(figure_output_folder, f"figure_{figure_number}_fsc_average_vs_iteration.png")

    assert os.path.exists(path_to_store_fsc_average_pickle), f"Path does not exist: {path_to_store_fsc_average_pickle}"
    assert os.path.exists(path_to_store_radial_profile_pickle), f"Path does not exist: {path_to_store_radial_profile_pickle}"

    # Load the data
    with open(path_to_store_fsc_average_pickle, "rb") as f:
        fsc_average_iterations = pickle.load(f)
    
    figsize_mm = (60,60)
    rcParams_new = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcParams_new):
        fig = pretty_lineplot_XY(\
            xdata=fsc_average_iterations.keys(), ydata=fsc_average_iterations.values(), \
            xlabel="Iteration", ylabel=r"$\langle$FSC$\rangle$", figsize_mm=figsize_mm, \
            marker="", markersize=12,fontscale=4,font="Helvetica"
        )
        # Change color to black
        fig.axes[0].lines[0].set_color("black")
        # Change the linewidth
        fig.axes[0].lines[0].set_linewidth(1)
        fig.axes[0].set_xticks([0,25,50])
        
        if SAVE_FIGURE:
            fig.savefig(fsc_average_figure_path, dpi=600, bbox_inches='tight')
            fig.savefig(fsc_average_figure_path_png, dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    # Print the start time
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {processing_time}")
    print("="*80)


    




    

